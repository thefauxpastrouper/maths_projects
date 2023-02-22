import numpy as np

def encode_message(message, encoding_matrix):
    """
    Encode a message using a nonsingular encoding matrix.
    """
    # Convert the message to a numpy array
    message_array = np.array(message, dtype=np.float64)
    
    # Encode the message by multiplying it by the encoding matrix
    encoded_message = np.dot(message_array, encoding_matrix)
    
    # Round the encoded message to integers and return as a list
    return encoded_message.round().astype(int).tolist()

def decode_message(encoded_message, decoding_matrix):
    """
    Decode an encoded message using the inverse of the encoding matrix.
    """
    # Convert the encoded message to a numpy array
    encoded_message_array = np.array(encoded_message, dtype=np.float64)
    
    # Decode the encoded message by multiplying it by the decoding matrix
    decoded_message = np.dot(encoded_message_array, decoding_matrix)
    
    # Round the decoded message to integers and return as a list
    return decoded_message.round().astype(int).tolist()
