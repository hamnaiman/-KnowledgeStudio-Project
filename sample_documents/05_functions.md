# Python Functions

A function is a reusable block of code defined using `def`. Its body runs when it is called. Parameters are names in the definition; arguments are values passed to the call. Default parameter values apply when the corresponding argument is omitted.

`return` sends a result to the caller and ends that function call. `print()` displays output; it does not replace returning a result. A function without an explicit return value returns None. Variables assigned inside a function are normally local to it.

```python
def add(a, b):
    return a + b

result = add(2, 3)
print(result)
```

Here `a` and `b` are parameters, 2 and 3 are arguments, and the returned value is 5. The caller stores 5 in `result`, then prints it.
