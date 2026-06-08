# HumanEval 4-shot generations: phi-1.5 vs C5-stage1 vs C5-final (10 examples)

4-shot prompting: HumanEval problems 0-3 prepended as worked examples (full prompt + canonical_solution). Targets = remaining 160 problems.

4-shot pass@1: phi-1.5 = 56/160 (35.0%); C5-stage1 = 9/160 (5.6%); C5-final = 11/160 (6.9%).

Same 10 HumanEval IDs as `c5_stage1_vs_phi1_humaneval_samples.md` (the 0-shot reference) for direct comparability.

---

## HumanEval/6

### Prompt

```python
from typing import List


def parse_nested_parens(paren_string: str) -> List[int]:
    """ Input to this function is a string represented multiple groups for nested parentheses separated by spaces.
    For each of the group, output the deepest level of nesting of parentheses.
    E.g. (()()) has maximum two levels of nesting while ((())) has three.

    >>> parse_nested_parens('(()()) ((())) () ((())()())')
    [2, 3, 1, 3]
    """
```

### Canonical solution

```python
    def parse_paren_group(s):
        depth = 0
        max_depth = 0
        for c in s:
            if c == '(':
                depth += 1
                max_depth = max(depth, max_depth)
            else:
                depth -= 1

        return max_depth

    return [parse_paren_group(x) for x in paren_string.split(' ') if x]
```

### phi-1.5 4-shot (pass: False)

```python
    result = []
    current_depth = 0
    current_level = 0

    for c in paren_string:
        if c == '(':
            current_depth += 1
            current_level += 1
            result.append(current_level)
        elif c == ')':
            current_depth -= 1
            current_level -= 1
            result.append(current_level)

    return result
```

### C5-stage1 4-shot (pass: False)

```python
    result = []
    current_string = []
    current_depth = 0

    for c in paren_string:
        if c == '(':
            current_depth += 1
            current_string.append(c)
        elif c == ')':
            current_depth -= 1
            current_string.append(c)

            if current_depth == 0:
                result.append(0)
                current_string.clear()

    return result
```

### C5-final 4-shot (pass: False)

```python
    result = []
    current_string = []
    current_depth = 0

    for c in paren_string:
        if c == '(':
            current_depth += 1
            current_string.append(c)
        elif c == ')':
            current_depth -= 1
            current_string.append(c)

            if current_depth == 0:
                result.append(current_string)
                current_string.clear()

    return result
```

---

## HumanEval/22

### Prompt

```python
from typing import List, Any


def filter_integers(values: List[Any]) -> List[int]:
    """ Filter given list of any python values only for integers
    >>> filter_integers(['a', 3.14, 5])
    [5]
    >>> filter_integers([1, 2, 3, 'abc', {}, []])
    [1, 2, 3]
    """
```

### Canonical solution

```python
    return [x for x in values if isinstance(x, int)]
```

### phi-1.5 4-shot (pass: True)

```python
    return [value for value in values if isinstance(value, int)]
```

### C5-stage1 4-shot (pass: False)

```python
    return [int(v) for v in values]
```

### C5-final 4-shot (pass: False)

```python
    return [int(v) for v in values if v]
```

---

## HumanEval/26

### Prompt

```python
from typing import List


def remove_duplicates(numbers: List[int]) -> List[int]:
    """ From a list of integers, remove all elements that occur more than once.
    Keep order of elements left the same as in the input.
    >>> remove_duplicates([1, 2, 3, 2, 4])
    [1, 3, 4]
    """
```

### Canonical solution

```python
    import collections
    c = collections.Counter(numbers)
    return [n for n in numbers if c[n] <= 1]
```

### phi-1.5 4-shot (pass: False)

```python
    seen = set()
    result = []

    for num in numbers:
        if num not in seen:
            seen.add(num)
            result.append(num)

    return result
```

### C5-stage1 4-shot (pass: False)

```python
    result = []
    for num in numbers:
        if num not in result:
            result.append(num)
    return result
```

### C5-final 4-shot (pass: False)

```python
    result = []
    for num in numbers:
        if num in result:
            continue
        else:
            result.append(num)
    return result
```

---

## HumanEval/28

### Prompt

```python
from typing import List


def concatenate(strings: List[str]) -> str:
    """ Concatenate list of strings into a single string
    >>> concatenate([])
    ''
    >>> concatenate(['a', 'b', 'c'])
    'abc'
    """
```

### Canonical solution

```python
    return ''.join(strings)
```

### phi-1.5 4-shot (pass: True)

```python
    return ''.join(strings)
```

### C5-stage1 4-shot (pass: True)

```python
    return ''.join(strings)
```

### C5-final 4-shot (pass: True)

```python
    return ''.join(strings)
```

---

## HumanEval/35

### Prompt

```python


def max_element(l: list):
    """Return maximum element in the list.
    >>> max_element([1, 2, 3])
    3
    >>> max_element([5, 3, -5, 2, -3, 3, 9, 0, 123, 1, -10])
    123
    """
```

### Canonical solution

```python
    m = l[0]
    for e in l:
        if e > m:
            m = e
    return m
```

### phi-1.5 4-shot (pass: True)

```python
    return max(l)
```

### C5-stage1 4-shot (pass: True)

```python
    return max(l)
```

### C5-final 4-shot (pass: True)

```python
    return max(l)
```

---

## HumanEval/57

### Prompt

```python


def monotonic(l: list):
    """Return True is list elements are monotonically increasing or decreasing.
    >>> monotonic([1, 2, 4, 20])
    True
    >>> monotonic([1, 20, 4, 10])
    False
    >>> monotonic([4, 1, 0, -10])
    True
    """
```

### Canonical solution

```python
    if l == sorted(l) or l == sorted(l, reverse=True):
        return True
    return False
```

### phi-1.5 4-shot (pass: True)

```python
    return all(l[i] <= l[i + 1] for i in range(len(l) - 1)) or all(
        l[i] >= l[i + 1] for i in range(len(l) - 1))
```

### C5-stage1 4-shot (pass: False)

```python
    if len(l) == 0:
        return True
    else:
        return l[0] > l[-1]
```

### C5-final 4-shot (pass: False)

```python
    return all(l[i] > l[i + 1] for i in range(len(l) - 1))
```

---

## HumanEval/62

### Prompt

```python


def derivative(xs: list):
    """ xs represent coefficients of a polynomial.
    xs[0] + xs[1] * x + xs[2] * x^2 + ....
     Return derivative of this polynomial in the same form.
    >>> derivative([3, 1, 2, 4, 5])
    [1, 4, 12, 20]
    >>> derivative([1, 2, 3])
    [2, 6]
    """
```

### Canonical solution

```python
    return [(i * x) for i, x in enumerate(xs)][1:]
```

### phi-1.5 4-shot (pass: True)

```python
    result = []
    for i in range(1, len(xs)):
        result.append(xs[i] * i)

    return result
```

### C5-stage1 4-shot (pass: False)

```python
    return [x * x for x in xs]
```

### C5-final 4-shot (pass: False)

```python
    return [x * x for x in xs]
```

---

## HumanEval/70

### Prompt

```python

def strange_sort_list(lst):
    '''
    Given list of integers, return list in strange order.
    Strange sorting, is when you start with the minimum value,
    then maximum of the remaining integers, then minimum and so on.

    Examples:
    strange_sort_list([1, 2, 3, 4]) == [1, 4, 2, 3]
    strange_sort_list([5, 5, 5, 5]) == [5, 5, 5, 5]
    strange_sort_list([]) == []
    '''
```

### Canonical solution

```python
    res, switch = [], True
    while lst:
        res.append(min(lst) if switch else max(lst))
        lst.remove(res[-1])
        switch = not switch
    return res
```

### phi-1.5 4-shot (pass: False)

```python
    if not lst:
        return []

    lst.sort()
    return [lst[0]] + [lst[i] for i in range(1, len(lst)) if lst[i]!= lst[0]] + [lst[-1]]
```

### C5-stage1 4-shot (pass: False)

```python
    if lst == []:
        return lst
    else:
        return sorted(lst, key=lambda x: x)
```

### C5-final 4-shot (pass: False)

```python
    lst.sort()
    return lst
```

---

## HumanEval/139

### Prompt

```python

def special_factorial(n):
    """The Brazilian factorial is defined as:
    brazilian_factorial(n) = n! * (n-1)! * (n-2)! * ... * 1!
    where n > 0

    For example:
    >>> special_factorial(4)
    288

    The function will receive an integer as input and should return the special
    factorial of this integer.
    """
```

### Canonical solution

```python
    fact_i = 1
    special_fact = 1
    for i in range(1, n+1):
        fact_i *= i
        special_fact *= fact_i
    return special_fact
```

### phi-1.5 4-shot (pass: False)

```python
    if n == 0:
        return 1
    else:
        return n * special_factorial(n-1)
```

### C5-stage1 4-shot (pass: False)

```python
    if n == 0:
        return 1
    else:
        return n * special_factorial(n-1) * special_factorial(n-2) * special_factorial(n-3)
```

### C5-final 4-shot (pass: False)

```python
    return n * (n - 1) * (n - 2) * (n - 3) * (n - 4)
```

---

## HumanEval/163

### Prompt

```python

def generate_integers(a, b):
    """
    Given two positive integers a and b, return the even digits between a
    and b, in ascending order.

    For example:
    generate_integers(2, 8) => [2, 4, 6, 8]
    generate_integers(8, 2) => [2, 4, 6, 8]
    generate_integers(10, 14) => []
    """
```

### Canonical solution

```python
    lower = max(2, min(a, b))
    upper = min(8, max(a, b))

    return [i for i in range(lower, upper+1) if i % 2 == 0]
```

### phi-1.5 4-shot (pass: False)

```python
    result = []
    for i in range(a, b + 1):
        if i % 2 == 0:
            result.append(i)

    return result
```

### C5-stage1 4-shot (pass: False)

```python
    return [int(a) for i in range(0, 10, 2)] + [int(b)]
```

### C5-final 4-shot (pass: False)

```python
    return [int(x) for x in str(a) + str(b)]
```

---
