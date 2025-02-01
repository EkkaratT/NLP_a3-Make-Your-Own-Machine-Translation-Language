# NLP_a3-Make-Your-Own-Machine-Translation-Language

# Machine Translation Between English and Thai

This assignment focuses on developing a machine translation model for translating between English and Thai. It utilizes attention mechanisms to improve translation quality and computational efficiency. The dataset is sourced from Hugging Face's **kvush/english_thai_texts**, and the model was trained using three different attention mechanisms: **General Attention**, **Multiplicative Attention**, and **Additive Attention**.

---

## Task 1: Get Language Pair

### Dataset Selection for Translation Between Native Language (Thai) and English

For the task of training a translation model between Thai and English, I used the dataset from the Hugging Face repository **kvush/english_thai_texts**, which provides a large collection of aligned English-Thai sentence pairs. This dataset is publicly available and can be accessed from Hugging Face's datasets library.

**Dataset Source:**
- Dataset: kvush/english_thai_texts
- Repository: Hugging Face ([Link](https://huggingface.co/datasets/kvush/english_thai_texts))

### Process of Preparing the Dataset for Translation Model

The dataset preparation involves several important steps to ensure that the text is in a usable format for training a translation model. Below is a detailed breakdown of the preparation process:

#### Step 1: Dataset Splitting
The dataset is split into three subsets:
- **Training Set:** The first 41,901 sentence pairs are used for training.
- **Validation Set:** The next 5,986 sentence pairs are used for validation.
- **Test Set:** The last 11,972 sentence pairs are reserved for testing.

#### Step 2: Tokenization
Tokenization is a crucial step in preparing text for neural machine translation. In this case:
- **English Tokenization:** Using the SpaCy tokenizer for English text. SpaCy is a well-known and efficient library for tokenizing English text. The tokenizer used is the 'en_core_web_sm' model from SpaCy.
- **Thai Tokenization:** For Thai text, tokenization is more complex due to the lack of spaces between words. Using the **pythainlp** library, which is specifically designed for tokenizing Thai text.

#### Step 3: Vocabulary Creation
A vocabulary for each language (English and Thai) is created based on the tokenized sentences in the training set. This vocabulary maps each unique token to a corresponding index. Special tokens, such as `<unk>`, `<pad>`, `<sos>`, and `<eos>`, are also included in the vocabulary.

#### Step 4: Text Normalization
In addition to tokenization, text normalization is essential to ensure consistency across the dataset. Common preprocessing includes:
- Lowercasing text (for English).
- Removing special characters or symbols that are irrelevant for translation.
- Stripping extra whitespace from the text.

#### Step 5: Sequence Padding and Transformation to Tensors
Since machine learning models typically require inputs to be of the same length, sequences of tokens are padded to the maximum length in a batch. This is handled by the `pad_sequence` function from PyTorch, which pads sequences with the `<pad>` token to ensure uniform length across batches.

#### Step 6: DataLoader Setup
Finally, the prepared data is wrapped into `DataLoader` objects for efficient batching and shuffling during training, validation, and testing.

---

## Task 2: Experiment with Attention Mechanisms

### General Attention
This is the simplest attention mechanism where the attention score is computed as the dot product of the decoder and encoder hidden states.
    Code:
            General Attention (dot product)
            energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale



### Multiplicative Attention
In this variation, the query (decoder hidden state) is transformed by a learnable weight matrix W before computing the attention score with the encoder hidden state.

### Additive Attention
In additive attention, the query and key are both transformed by learnable weight matrices, and their results are combined using the tanh activation function. The final attention score is produced by applying a weight vector v.

---

## Task 3: Evaluation and Verification

### Compare the performance of these attention mechanisms:

| Attentions            | Training Loss | Training PPL | Validation Loss | Validation PPL |
|-----------------------|---------------|--------------|-----------------|----------------|
| **General Attention**  | 1.052         | 2.863        | 2.067           | 7.897          |
| **Multiplicative Attention** | 1.021         | 2.777        | 2.018           | 7.525          |
| **Additive Attention** | 1.108         | 3.027        | 2.148           | 8.565          |

### Translation Accuracy:
- **Multiplicative Attention** performs the best in terms of translation accuracy with the lowest perplexity (7.580) on the test set.
- **General Attention** is decent, with a test perplexity of 7.826, showing reasonable accuracy but not as good as Multiplicative.
- **Additive Attention** has the highest test perplexity (8.489), indicating lower translation accuracy.

### Computational Efficiency:
- **General Attention** is the fastest, taking 0m 50s per epoch, making it the most computationally efficient.
- **Multiplicative Attention** takes 0m 53s per epoch, slightly slower than General Attention but still relatively efficient.
- **Additive Attention** takes 0m 59s per epoch, which is the slowest of the three attention mechanisms. (Running on same device: puffer)

### Other Relevant Metrics:
- **General Attention** strikes the best balance between translation accuracy and computational efficiency. It achieves relatively high performance with faster training time, making it suitable for scenarios requiring quick model iterations.
- **Multiplicative Attention**, while slightly slower, has a slightly better translation accuracy than Additive Attention, making it a strong candidate when translation quality is prioritized over training speed.
- **Additive Attention**, although slower and slightly less accurate, still demonstrates reasonable translation performance, though it could be less favorable when computational resources and time are limited.

---

## Task 4: Machine Translation - Web Application Development

This web application allows users to input text in English and receive a translation in Thai. The translation is generated using a machine translation model.

### How the App Works:
1. **User Input:** The user types a sentence in English into the input box on the webpage.
2. **Translation Process:**
   - The text is sent to the backend, where a pre-trained machine translation model processes the input.
   - The model translates the sentence from English to Thai.
3. **Output Display:** The translated Thai sentence is then displayed on the webpage.

---

**Note:** Attention maps and training plots will be added here later for a better understanding of the model's performance and attention behavior during translation.

---

Feel free to check out the code and experiment with various attention mechanisms to see how they perform for your translation tasks.
