# Conditional Statements

Conditional statements control which block of code executes. An `if` block runs when its condition is True. An `elif` condition is checked when all earlier conditions in the chain were False. An `else` block runs when no earlier condition matched. Only the first matching branch of an if/elif/else chain executes.

Use `==`, `!=`, `<`, `>`, `<=` and `>=` for comparisons. `and` requires both conditions to be True; `or` requires at least one; `not` reverses a Boolean value. End the condition line with a colon and indent the block.

```python
marks = 72
if marks >= 80:
    print('Excellent')
elif marks >= 50:
    print('Pass')
else:
    print('Needs practice')
```
