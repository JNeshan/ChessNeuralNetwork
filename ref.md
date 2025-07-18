
## Headers
```
# H1 - Project Title
## H2 - Major Sections
### H3 - Subsections
#### H4 - Details
##### H5 - Minor Points
###### H6 - Smallest
```

## Text Formatting
```
**bold text**
*italic text*
`inline code`
~~strikethrough~~
> blockquote
```

## Lists
```
### Unordered
- Item 1
- Item 2
  - Nested item
  - Another nested

### Ordered
1. First item
2. Second item
   1. Nested numbered
   2. Another nested

### Task Lists
- [x] Completed task
- [ ] Incomplete task
```

## Links and Images
```
[Link text](https://example.com)
[Relative link](./docs/readme.md)
![Image alt text](path/to/image.png)
![Image with size](image.png "Title text")
```

## Tables
```
| Column 1 | Column 2 | Column 3 |
|----------|----------|----------|
| Data 1   | Data 2   | Data 3   |
| Value A  | Value B  | Value C  |

| Left | Center | Right |
|:-----|:------:|------:|
| L1   |   C1   |    R1 |
```

## Mathematical Equations
```
### Inline Math
The function $f(x) = x^2$ represents...

### Block Math
$$
\sum_{i=1}^{n} x_i = x_1 + x_2 + ... + x_n
$$

### Common Symbols
- Summation: $\sum_{i=1}^{n}$
- Integration: $\int_{a}^{b}$
- Derivative: $\frac{d}{dx}$, $\frac{\partial}{\partial x}$
- Fractions: $\frac{a}{b}$
- Subscript: $x_i$, $x_{ij}$
- Superscript: $x^2$, $x^{n+1}$
- Greek letters: $\alpha$, $\beta$, $\gamma$, $\theta$, $\sigma$, $\mu$
- Matrices: $\begin{bmatrix} a & b \\ c & d \end{bmatrix}$
- Vectors: $\vec{v}$, $\mathbf{v}$
```

## Code Documentation
```
### Function Documentation
```cpp
/**
 * @brief Computes forward pass
 * @param input Input tensor
 * @return Output tensor
 */
Tensor forward(const Tensor& input);
```

### API Reference
#### `className::methodName()`
**Parameters:**
- `param1` (type): Description
- `param2` (type): Description

**Returns:** Description of return value

**Example:**
```cpp
Tensor result = layer.forward(input);
```
```

## Project Structure
```
### File Trees
```
project/
├── src/
│   ├── main.cpp
│   └── utils.cpp
├── include/
│   └── headers.h
└── README.md
```

### Directory Links
- [Source Code](./src/)
- [Documentation](./docs/)
- [Tests](./tests/)
```

## Notes and Callouts
```
> **Note:** Important information
> 
> **Warning:** Potential issues
> 
> **TODO:** Items to complete

### Collapsible Sections
<details>
<summary>Click to expand</summary>

Hidden content here
</details>
```

## Neural Network Specific Math
```
### Forward Pass
$$
y = f(Wx + b)
$$

### Backpropagation
$$
\frac{\partial L}{\partial W} = \frac{\partial L}{\partial y} \cdot \frac{\partial y}{\partial W}
$$

### Convolution
$$
(f * g)[n] = \sum_{m=-\infty}^{\infty} f[m] \cdot g[n-m]
$$

### Softmax
$$
\text{softmax}(x_i) = \frac{e^{x_i}}{\sum_{j=1}^{K} e^{x_j}}
$$

### Loss Functions
$$
\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y_i})^2
$$

$$
\text{Cross-entropy} = -\sum_{i=1}^{n} y_i \log(\hat{y_i})
$$