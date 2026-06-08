# HumanEval generations: C5-stage1 vs phi-1 (10 randomly selected)

Random seed 42; problems sampled from openai_humaneval test set (164 total).

C5-stage1: trained on 15.4B tokens of multi-language Stack + markup (~1.5B Python).
phi-1: trained on ~7B filtered Stack-Python + synthetic textbooks, 50B token budget (8 epochs).

---

## HumanEval/6

### Prompt (given to both models)

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

### Reference (canonical) solution

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

### C5-stage1 generation

```python
(empty — model gave up after docstring)
```

### phi-1 generation

```python
    stack = []
    max_depth = 0
    current_depth = 0
    for char in paren_string:
        if char == '(':
            stack.append(current_depth)
            current_depth += 1
            max_depth = max(max_depth, current_depth)
        elif char == ')':
            stack.pop()
            current_depth -= 1
    return [max_depth - i for i in stack]
```

---

## HumanEval/22

### Prompt (given to both models)

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

### Reference (canonical) solution

```python
    return [x for x in values if isinstance(x, int)]

```

### C5-stage1 generation

```python
(empty — model gave up after docstring)
```

### phi-1 generation

```python
    return [value for value in values if isinstance(value, int)]
```

---

## HumanEval/26

### Prompt (given to both models)

```python
from typing import List


def remove_duplicates(numbers: List[int]) -> List[int]:
    """ From a list of integers, remove all elements that occur more than once.
    Keep order of elements left the same as in the input.
    >>> remove_duplicates([1, 2, 3, 2, 4])
    [1, 3, 4]
    """

```

### Reference (canonical) solution

```python
    import collections
    c = collections.Counter(numbers)
    return [n for n in numbers if c[n] <= 1]

```

### C5-stage1 generation

```python
(empty — model gave up after docstring)
```

### phi-1 generation

```python
    result = []
    for num in numbers:
        if num not in result:
            result.append(num)
    return result
```

---

## HumanEval/28

### Prompt (given to both models)

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

### Reference (canonical) solution

```python
    return ''.join(strings)

```

### C5-stage1 generation

```python
(empty — model gave up after docstring)
```

### phi-1 generation

```python
    return ''.join(strings)
```

---

## HumanEval/35

### Prompt (given to both models)

```python


def max_element(l: list):
    """Return maximum element in the list.
    >>> max_element([1, 2, 3])
    3
    >>> max_element([5, 3, -5, 2, -3, 3, 9, 0, 123, 1, -10])
    123
    """

```

### Reference (canonical) solution

```python
    m = l[0]
    for e in l:
        if e > m:
            m = e
    return m

```

### C5-stage1 generation

```python
def max_element(l: list):
    """Return maximum element in the list.
    >>> max_element([1, 2, 3])
    3
    >>> max_element([5, 3, -5, 2, -3, 3, 9, 0, 123, 1, -10])
    123
    """
```

### phi-1 generation

```python
def max_element(l: list):
    """Return maximum element in the list.
    >>> max_element([1, 2, 3])
    3
    >>> max_element([5, 3, -5, 2, -3, 3, 9, 0, 123, 1, -10])
    123
    """
    max_val = l[0]
    for i in range(1, len(l)):
        if l[i] > max_val:
            max_val = l[i]
    return max_val
```

---

## HumanEval/57

### Prompt (given to both models)

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

### Reference (canonical) solution

```python
    if l == sorted(l) or l == sorted(l, reverse=True):
        return True
    return False

```

### C5-stage1 generation

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

### phi-1 generation

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
    if len(l) < 2:
        return True
    increasing = decreasing = True
    for i in range(1, len(l)):
        if l[i] > l[i-1]:
            decreasing = False
        elif l[i] < l[i-1]:
            increasing = False
        if not increasing and not decreasing:
            return False
    return True
```

---

## HumanEval/62

### Prompt (given to both models)

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

### Reference (canonical) solution

```python
    return [(i * x) for i, x in enumerate(xs)][1:]

```

### C5-stage1 generation

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


    def __init__(self, x):
        self.x = x

    def __str__(self):
        return str(self.x)

    def __repr__(self):
        return str(self.x)

    def __add__(self, other):
        return Polynomial(self.x + other.x)

    def __radd__(self, other):
        return Polynomial(self.x + other.x)

    def __mul__(self, other):
        return Polynomial(self.x * other.x)

    def __rmul__(self, other):
        return Polynomial(self.x * other.x)

    def __truediv__(self, other):
        return Polynomial(self.x / other.x)

    def __rtruediv__(self, other):
        return Polynomial(self.x / other.x)

    def __pow__(self, other):
        return Polynomial(self.x ** other)

    def __rmul__(self, other):
        return Polynomial(self.x * other)

    def __pow__(self, other):
        return Polynomial(self.x ** other)

    def __pow__(self, other):
        return Polynomial(self.x ** other)

    def __pow__(self, other):
        return Polynomial(self.x ** other)

    def __pow__(self, other):
        return Polynomial(self.x ** other)

    def __pow__(self, other):
        return Polynomial(self.x ** other)

    def __pow__(self, other):
        return Polynomial(self.x ** other)

    def __pow__(self, other):
        return Polynomial(self.x ** other)

    def __pow__(self, other):
        return Polynomial(self.x ** other)

    def __pow__(self, other):
        return Polynomial(self.x ** other)

    def __pow__(self, other):
        return Polynomial(self.x ** other)

    def __pow__(self, other):
        return Polynomial(self.x ** other)

    def __pow__(self, other):
        return Polynomial(self.x ** other)

    def __pow__(self, other):
        return Polynomial(self
```

### phi-1 generation

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
    if len(xs) == 1:
        return [0]
    else:
        return [xs[i] * (len(xs) - i - 1) for i in range(len(xs) - 1)]
```

---

## HumanEval/70

### Prompt (given to both models)

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

### Reference (canonical) solution

```python
    res, switch = [], True
    while lst:
        res.append(min(lst) if switch else max(lst))
        lst.remove(res[-1])
        switch = not switch
    return res

```

### C5-stage1 generation

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


    # Sort the list
    lst.sort()

    # Return the list
    return lst
```

### phi-1 generation

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
    while lst:
        min_val = min(lst)
        max_val = max(lst)
        lst.remove(min_val)
        lst.remove(max_val)
        lst.append(min_val)
        lst.append(max_val)
    return lst
```

---

## HumanEval/139

### Prompt (given to both models)

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

### Reference (canonical) solution

```python
    fact_i = 1
    special_fact = 1
    for i in range(1, n+1):
        fact_i *= i
        special_fact *= fact_i
    return special_fact

```

### C5-stage1 generation

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


    if n == 0:
        return 1
    elif n == 1:
        return 1
    elif n == 2:
        return 1
    elif n == 3:
        return 1
    elif n == 4:
        return 1
    else:
        return 1 * (n-1) * (n-2) * (n-3) * (n-4)
```

### phi-1 generation

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
    if n == 0:
        return 1
    else:
        return n * special_factorial(n-1)
```

---

## HumanEval/163

### Prompt (given to both models)

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

### Reference (canonical) solution

```python
    lower = max(2, min(a, b))
    upper = min(8, max(a, b))

    return [i for i in range(lower, upper+1) if i % 2 == 0]

```

### C5-stage1 generation

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


    def get_even_digits(a, b):
        """
        Given two positive integers a and b, return the even digits between a
        and b, in ascending order.

        For example:
        generate_integers(2, 8) => [2, 4, 6, 8]
        generate_integers(8, 2) => [2, 4, 6, 8]
        generate_integers(10, 14) => []
        """
        return [int(a) % 2, int(b) % 2]

    return get_even_digits(a, b)
```

### phi-1 generation

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
    result = []
    for num in range(a, b+1):
        for digit in str(num):
            if int(digit) % 2 == 0:
                result.append(int(digit))
    return result
```

---

