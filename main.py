import json
import os
from functools import lru_cache
from io import BytesIO
from pathlib import Path
from threading import Lock
from typing import Literal
from uuid import uuid4

import chromadb
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from openai import APIError, OpenAI
from pydantic import BaseModel, Field
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

from prompts import TEMPLATES

BASE = Path(__file__).resolve().parent
load_dotenv(BASE / '.env', override=True)
DATA = BASE / 'data'
DATA.mkdir(exist_ok=True)
db = chromadb.PersistentClient(path=str(DATA / 'chroma_db'))
POINTER = DATA / 'active.json'
lock = Lock()
app = FastAPI(title='AI Knowledge Assistant')


@lru_cache(maxsize=1)
def encoder():
    return SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device='cpu')


def current():
    if not POINTER.exists():
        return None
    name = json.loads(POINTER.read_text())['collection']
    return db.get_collection(name, embedding_function=None)


def chunks(text):
    tokenizer = encoder().tokenizer
    tokens = tokenizer.encode(text, add_special_tokens=False)
    return [tokenizer.decode(tokens[i:i + 160], skip_special_tokens=True)
            for i in range(0, len(tokens), 128)]


def retrieve(question):
    with lock:
        saved = current()
        if saved is None or saved.count() < 3:
            raise HTTPException(400, 'Upload at least five documents first.')
        try:
            vector = encoder().encode([question], normalize_embeddings=True).tolist()
            found = saved.query(query_embeddings=vector, n_results=3,
                                include=['documents', 'metadatas', 'distances'])
        except Exception:
            raise HTTPException(500, 'Search failed. Check model availability or upload the documents again.')
        sources = []
        for text, info, distance in zip(found['documents'][0], found['metadatas'][0], found['distances'][0]):
            sources.append({'text': text, 'file': info['file'], 'page': info['page'],
                            'distance': round(float(distance), 4),
                            'similarity': round(1 - float(distance), 4)})
        return sources, saved.name


def generate(question, sources, technique):
    key = os.getenv('GROQ_API_KEY', '').strip()
    if not key:
        raise HTTPException(503, 'Add GROQ_API_KEY in .env and restart the app.')
    context = '\n\n'.join(f"[{i}] {s['file']} / page {s['page']}\n{s['text']}"
                            for i, s in enumerate(sources, 1))
    model = os.getenv('GROQ_MODEL', '').strip() or 'openai/gpt-oss-20b'
    try:
        with OpenAI(api_key=key, base_url='https://api.groq.com/openai/v1',
                    timeout=90, max_retries=0) as client:
            result = client.chat.completions.create(
                model=model, temperature=0, max_completion_tokens=3000,
                messages=[{'role': 'system', 'content': TEMPLATES[technique]},
                          {'role': 'user', 'content': f'Actual context:\n{context}\n\nQuestion: {question}'}])
    except APIError as error:
        code = getattr(error, 'status_code', None)
        message = {401: 'Invalid Groq API key.', 403: 'Model access is not enabled.',
                   404: 'Model not found. Check GROQ_MODEL.',
                   429: 'Groq rate limit reached. Wait, then retry.',
                   400: 'Groq rejected the request. Check model/account settings.'}
        raise HTTPException(502, message.get(code, 'Cannot reach Groq. Check connection or retry later.'))
    answer = result.choices[0].message.content if result.choices else None
    if not answer or not answer.strip():
        raise HTTPException(502, 'The model returned no answer. Please retry.')
    return answer, model


@app.get('/')
def home():
    return FileResponse(BASE / 'index.html')


@app.get('/status')
def status():
    with lock:
        saved = current()
        info = saved.metadata if saved else {}
        return {'files': json.loads(info.get('files', '[]')), 'chunks': saved.count() if saved else 0,
                'dataset_id': saved.name if saved else '',
                'groq_ready': bool(os.getenv('GROQ_API_KEY', '').strip()),
                'model': os.getenv('GROQ_MODEL', '').strip() or 'openai/gpt-oss-20b'}


@app.get('/benchmark')
def benchmark():
    return json.loads((BASE / 'benchmark.json').read_text(encoding='utf-8'))


@app.get('/templates')
def templates():
    return TEMPLATES


@app.post('/upload')
def upload(files: list[UploadFile] = File(...)):
    if not 5 <= len(files) <= 20:
        raise HTTPException(400, 'Select between 5 and 20 documents.')
    names = [Path(file.filename or '').name for file in files]
    if len(set(names)) != len(names):
        raise HTTPException(400, 'Use different filenames for each document.')
    pages = []
    total = 0
    for file, name in zip(files, names):
        suffix = Path(name).suffix.lower()
        if suffix not in ['.pdf', '.txt', '.md']:
            raise HTTPException(400, 'Only PDF, TXT and Markdown (.md) files are supported.')
        content = file.file.read(5 * 1024 * 1024 + 1)
        total += len(content)
        if len(content) > 5 * 1024 * 1024 or total > 25 * 1024 * 1024:
            raise HTTPException(400, 'Use files up to 5 MB each and 25 MB total.')
        try:
            texts = ([page.extract_text() or '' for page in PdfReader(BytesIO(content)).pages]
                     if suffix == '.pdf' else [content.decode('utf-8-sig')])
        except Exception:
            raise HTTPException(400, f'Cannot read {name}. Use an unlocked text PDF or UTF-8 text file.')
        if not any(text.strip() for text in texts):
            raise HTTPException(400, f'{name} has no extractable text. Scanned PDFs need OCR.')
        pages.extend((name, number, text) for number, text in enumerate(texts, 1) if text.strip())
    with lock:
        name = 'docs_' + uuid4().hex
        try:
            documents, metadata = [], []
            for filename, page, text in pages:
                for chunk in chunks(text):
                    if chunk.strip():
                        documents.append(chunk)
                        metadata.append({'file': filename, 'page': page})
            saved = db.create_collection(name, embedding_function=None,
                    metadata={'hnsw:space': 'cosine', 'files': json.dumps(names)})
            for start in range(0, len(documents), 64):
                batch = documents[start:start + 64]
                vectors = encoder().encode(batch, normalize_embeddings=True).tolist()
                saved.add(ids=[str(i) for i in range(start, start + len(batch))],
                          documents=batch, metadatas=metadata[start:start + 64], embeddings=vectors)
            temporary = DATA / 'active.tmp'
            temporary.write_text(json.dumps({'collection': name}))
            temporary.replace(POINTER)
        except Exception:
            try:
                db.delete_collection(name)
            except Exception:
                pass
            raise HTTPException(500, 'Indexing failed. Check internet for the initial model download, memory and disk space.')
    return {'files': names, 'chunks': len(documents), 'dataset_id': name}


class Search(BaseModel):
    question: str = Field(min_length=1, max_length=1200)


class Ask(Search):
    technique: Literal['zero_shot', 'few_shot', 'role_based'] = 'zero_shot'


def clean(question):
    text = question.strip()
    if not text:
        raise HTTPException(400, 'Enter a question.')
    return text


@app.post('/search')
def search(body: Search):
    sources, dataset = retrieve(clean(body.question))
    return {'sources': sources, 'dataset_id': dataset}


@app.post('/ask')
def ask(body: Ask):
    question = clean(body.question)
    sources, dataset = retrieve(question)
    answer, model = generate(question, sources, body.technique)
    return {'answer': answer, 'sources': sources, 'technique': body.technique,
            'dataset_id': dataset, 'model': model}


@app.post('/compare')
def compare(body: Search):
    question = clean(body.question)
    if not os.getenv('GROQ_API_KEY', '').strip():
        raise HTTPException(503, 'Add GROQ_API_KEY in .env and restart the app.')
    sources, dataset = retrieve(question)
    answers = []
    for technique in TEMPLATES:
        try:
            answer, model = generate(question, sources, technique)
            answers.append({'technique': technique, 'answer': answer, 'model': model})
        except HTTPException as error:
            answers.append({'technique': technique, 'error': error.detail})
    return {'question': question, 'sources': sources, 'answers': answers, 'dataset_id': dataset}
