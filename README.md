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

#### Translation Accuracy:
- **Multiplicative Attention** performs the best in terms of translation accuracy with the lowest perplexity (7.580) on the test set.
- **General Attention** is decent, with a test perplexity of 7.826, showing reasonable accuracy but not as good as Multiplicative.
- **Additive Attention** has the highest test perplexity (8.489), indicating lower translation accuracy.

#### Computational Efficiency:
- **General Attention** is the fastest, taking 0m 50s per epoch, making it the most computationally efficient.
- **Multiplicative Attention** takes 0m 53s per epoch, slightly slower than General Attention but still relatively efficient.
- **Additive Attention** takes 0m 59s per epoch, which is the slowest of the three attention mechanisms. (Running on same device: puffer)

#### Other Relevant Metrics:
- **General Attention** strikes the best balance between translation accuracy and computational efficiency. It achieves relatively high performance with faster training time, making it suitable for scenarios requiring quick model iterations.
- **Multiplicative Attention**, while slightly slower, has a slightly better translation accuracy than Additive Attention, making it a strong candidate when translation quality is prioritized over training speed.
- **Additive Attention**, although slower and slightly less accurate, still demonstrates reasonable translation performance, though it could be less favorable when computational resources and time are limited.

#### Plots that show training and validation loss for each type of attention mechanism.
- **General attention**
  
  ![image](https://github.com/user-attachments/assets/0c7e4ee4-49f4-47ee-ab31-023c9a10b32a)


- **Multiplicative attention**
  
  ![image](https://github.com/user-attachments/assets/22c3cdef-30e1-46b3-80ab-8ed1d90b8046)

  
- **Additive attention**
  
  ![image](https://github.com/user-attachments/assets/2d4760d7-0a16-41e7-80a1-3b6f59506f04)


### Display the attention maps generated by your model
Attention maps are crucial for understanding how a model focuses on different parts of the input sequence when generating a translation, offering transparency and helping with model debugging and improvement.
- **General attention**
  
  ![image](https://github.com/user-attachments/assets/474043a3-d5ee-488f-aff4-2c8801930914)


- **Multiplicative attention**
  
  ![image](https://github.com/user-attachments/assets/37ae1cb1-52d0-4762-8779-2a6a8b4179af)


- **Additive attention**
  
  ![image](https://github.com/user-attachments/assets/7a05f007-2324-4ec0-a8a5-04b8e291b7c5)

#### Visual Representation:
	The attention map is visualized as a heatmap, where each cell corresponds to the attention score between an input token (in the source sequence) and an output token (in the target sequence).
	The attention scores are represented by color intensity: lower values appear darker (black), while higher values appear lighter (ranging from grey to white).

###	Effectiveness of Attention Mechanisms
The effectiveness of attention mechanisms in translating between languages English and Thai, can be evaluated based on translation accuracy, computational efficiency, and attention map interpretability.
	**General Attention** provides the best balance between translation accuracy and computational efficiency. It effectively captures complex dependencies in source and target languages, producing high-quality translations. The attention maps also show varied focus across the input sequence, aligning well with semantic relationships in the translation.
	**Multiplicative Attention** offers strong translation quality by capturing subtle token relationships but is more computationally expensive. This mechanism is ideal for generating accurate translations when resources are not a limiting factor, but it may not be as efficient for large-scale tasks.
	**Additive Attention** is the most computationally efficient, requiring less training time per epoch. However, it sometimes sacrifices translation quality, particularly for complex sentences. It is better suited for simpler tasks or situations where training time is crucial.

**General Attention** is the most effective overall, striking a good balance between accuracy and efficiency, making it the optimal choice for translating between languages with different syntactic structures.

---

## Task 4: Machine Translation - Web Application Development

This web application allows users to input text in English and receive a translation in Thai. The translation is generated using a machine translation model.

### How the App Works:
1. **User Input:** The user types a sentence in English into the input box on the webpage.
2. **Translation Process:**
   - The text is sent to the backend, where a pre-trained machine translation model processes the input.
   - The model translates the sentence from English to Thai.
3. **Output Display:** The translated Thai sentence is then displayed on the webpage.

![image](https://github.com/user-attachments/assets/ddf83027-ab0f-4246-9c10-4cfb09159cd3)
![image](https://github.com/user-attachments/assets/a5f044b5-2088-4c30-8c85-de6ce2e9302c)
![image](https://github.com/user-attachments/assets/17a60b07-a67f-4b30-9ac6-1706ce499dd8)
![image](https://github.com/user-attachments/assets/090dc6bd-b91b-4671-bb7f-48b6481eae1e)


