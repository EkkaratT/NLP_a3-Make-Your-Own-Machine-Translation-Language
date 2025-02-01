import torch
import torch.nn as nn
from flask import Flask, render_template, request
from models.classes import Encoder, Decoder, Seq2SeqTransformer  # Import from your classes.py file

# Define the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the token and vocab transforms
vocab_transform = torch.load('vocab_transform.pth')  # Load the entire vocab_transform dictionary
token_transform = torch.load('token_transform.pth')
ENG_LANGUAGE = 'input_text'
THAI_LANGUAGE = 'translated_text'

# Special symbols
UNK_IDX, PAD_IDX, SOS_IDX, EOS_IDX = 0, 1, 2, 3
special_symbols = ['<unk>', '<pad>', '<sos>', '<eos>']

# Define model architecture parameters
input_dim = len(vocab_transform[ENG_LANGUAGE])  # Assuming vocab is available for input language
output_dim = len(vocab_transform[THAI_LANGUAGE])  # Assuming vocab is available for output language
hid_dim = 256  # Hidden dimension updated to 256
enc_layers = 3  # Number of encoder layers
dec_layers = 3  # Number of decoder layers
enc_heads = 8  # Number of attention heads for the encoder
dec_heads = 8  # Number of attention heads for the decoder
enc_pf_dim = 512  # Feedforward dimension for encoder
dec_pf_dim = 512  # Feedforward dimension for decoder
enc_dropout = 0.1  # Dropout for encoder layers
dec_dropout = 0.1  # Dropout for decoder layers

# Function to load the model architecture
def create_model(input_dim, output_dim, hid_dim, enc_layers, dec_layers, enc_heads, dec_heads, enc_pf_dim, dec_pf_dim, enc_dropout, dec_dropout, device):
    encoder = Encoder(input_dim, hid_dim, enc_layers, enc_heads, enc_pf_dim, enc_dropout, device)
    decoder = Decoder(output_dim, hid_dim, dec_layers, dec_heads, dec_pf_dim, dec_dropout, device)
    model = Seq2SeqTransformer(encoder, decoder, src_pad_idx=PAD_IDX, trg_pad_idx=PAD_IDX, device=device)
    return model

# Instantiate the model
model = create_model(input_dim, output_dim, hid_dim, enc_layers, dec_layers, enc_heads, dec_heads, enc_pf_dim, dec_pf_dim, enc_dropout, dec_dropout, device)

# Load the pre-trained model weights (state_dict)
model.load_state_dict(torch.load('models/Seq2SeqTransformer.pt', map_location=device), strict=False)
model.to(device)
model.eval()  # Set the model to evaluation mode

# Helper function to create mask for padding
def create_mask(tensor, pad_idx):
    return (tensor != pad_idx).unsqueeze(1).unsqueeze(2)

# Text transformation for tokenization and numericalization
def sequential_transforms(*transforms):
    def func(txt_input):
        for transform in transforms:
            txt_input = transform(txt_input)
        return torch.tensor(txt_input, dtype=torch.long)  # Ensure it returns a tensor
    return func

# Construct text transformation for both languages
text_transform = {}
for ln in [ENG_LANGUAGE, THAI_LANGUAGE]:
    text_transform[ln] = sequential_transforms(
        token_transform[ln],  # Tokenization
        vocab_transform[ln],  # Numericalization
        lambda x: x           # Add EOS/BOS token if needed (or other transformations)
    )

# Translate function
# Translate function
def translate(input_text):
    # Transform the input text into a tensor of token indices
    input_tensor = text_transform[ENG_LANGUAGE](input_text).unsqueeze(0).to(device)  # Add batch dimension

    # Initialize target sequence with SOS token
    target_tensor = torch.tensor([SOS_IDX]).unsqueeze(0).to(device)

    # Translate
    with torch.no_grad():
        for _ in range(50):  # Maximum 50 tokens
            # Create source and target masks
            src_mask = create_mask(input_tensor, PAD_IDX)
            trg_mask = create_mask(target_tensor, PAD_IDX) & torch.tril(torch.ones((target_tensor.shape[1], target_tensor.shape[1]), device=device)).bool()
            
            # Forward pass through the model
            output, _ = model(input_tensor, target_tensor)

            # Get predicted next token
            next_token = output.argmax(2)[:, -1]

            # Append predicted token to target sequence
            target_tensor = torch.cat((target_tensor, next_token.unsqueeze(1)), dim=1)

            if next_token.item() == EOS_IDX:  # Stop when EOS token is predicted
                break

    # Convert token indices to words
    translated_tokens = target_tensor.squeeze(0).cpu().numpy()

    # Filter out the special tokens <sos> (SOS_IDX) and <eos> (EOS_IDX)
    translated_words = [vocab_transform[THAI_LANGUAGE].get_itos()[i] for i in translated_tokens if i not in [SOS_IDX, EOS_IDX]]

    # Output the translation
    translated_text = ' '.join(translated_words)
    return translated_text


# Flask app for the web interface
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/translate', methods=['POST'])
def translate_text():
    if request.method == 'POST':
        # Ensure 'input_text' exists in the form data
        try:
            input_text = request.form['input_text']
            translated_text = translate(input_text)
            return render_template('index.html', original_text=input_text, translated_text=translated_text)
        except KeyError:
            return "Error: Missing input text", 400

if __name__ == '__main__':
    app.run(debug=True)
