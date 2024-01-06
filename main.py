import heapq
from collections import defaultdict
from bitarray import bitarray
import numpy as np
import ast
import os
import hashlib
from typing import Union
from io import BytesIO
import io
import os
import matplotlib.pyplot as plt


def md5_hash(data):
    md5 = hashlib.md5()
    md5.update(data)
    return md5.hexdigest()

class Huffman:
    def __init__(self):
        pass

    @staticmethod
    def huffman_algorithm(dictionary):
        # Run the Huffman algorithm
        priority_queue = Huffman.dict_to_priority_queue(dictionary)
        huffman_tree = Huffman.build_tree(priority_queue)
        codes = Huffman.build_code_dictionary(huffman_tree)

        return codes

    @staticmethod
    def dict_to_priority_queue(dct):
        # Convert the given dictionary to a priority queue
        queue = [[weight, [key, ""]] for key, weight in dct.items()]
        heapq.heapify(queue)  # Sort by probability
        return queue

    @staticmethod
    def build_tree(queue):
        while len(queue) > 1:
            lo = heapq.heappop(queue) # Poping the element with lowest probability
            hi = heapq.heappop(queue) # Poping the element with highest probability
            for pair in lo[1:]:
                pair[1] = '1' + pair[1]  
            for pair in hi[1:]:
                pair[1] = '0' + pair[1]  
            # Combining signs with lowest probability, and add codes.
             # [lo[0] + hi[0]] asigns new probability , + lo[1:] + hi[1:] combines 2 signs
            heapq.heappush(queue, [lo[0] + hi[0]] + lo[1:] + hi[1:]) 
        return queue[0]

    @staticmethod
    def build_code_dictionary(tree):
        codes = defaultdict(str)
        for pair in tree[1:]:
            char, code = pair
            codes[char] = code
        return dict(codes)

    @staticmethod
    def calculate_frequencies(file_or_path: Union[str, BytesIO], tuple_size: int):
        freqs = defaultdict(int)
        N_padded_signs = 0
        
        # If a file path was given, open the file
        if isinstance(file_or_path, str):
            file = open(file_or_path, 'rb')
        else: # If a BytesIO object was given, use it directly
            file = file_or_path

        try:
            byte = file.read(tuple_size)
            while byte:
                if (len(byte) != tuple_size) & (len(byte) != 0):
                    N_padded_signs = (tuple_size - len(byte))
                    byte += (N_padded_signs) * (b'\x00')
                freqs[byte] += 1
                byte = file.read(tuple_size)
        finally:
            if isinstance(file_or_path, str): # Close the file only if it was opened in this function
                file.close()
        return freqs, N_padded_signs

    @staticmethod
    def calculate_probabilities(freqs):
        # Calculates the probabilities of bytes in a file.
        # Returns the dictionary
        total = sum(freqs.values())
        probabilities = {key: freq / total for key, freq in freqs.items()}
        return probabilities
    
    @staticmethod
    def to_bytearray(bit_string):
        padding_needed = (8 - len(bit_string) % 8) % 8
        padded_bit_string = bit_string + '0' * padding_needed
        return padded_bit_string, padding_needed

    @staticmethod
    def encode_file(filename_input, filename_output, tuple_size=1):
        freqs, N_padded_signs = Huffman.calculate_frequencies(filename_input, tuple_size)
        probabilities = Huffman.calculate_probabilities(freqs)
        huffman_codes = Huffman.huffman_algorithm(probabilities)
        
        # Encoding the huffman table with 1. order huffmans code.
        huffman_codes_bytes = str(huffman_codes).encode('utf-8')
        file = io.BytesIO(huffman_codes_bytes)
        freqs_, _ = Huffman.calculate_frequencies(file, 1)
        probabilities_ = Huffman.calculate_probabilities(freqs_)
        huffman_codes_of_huffman_codes = Huffman.huffman_algorithm(probabilities_)
        
        # Coding file with respect to the coding table
        coded_file_ = ''

        # Create a BytesIO object from the Huffman codes
        file = io.BytesIO(str(huffman_codes).encode('utf-8'))

        # Read from the BytesIO object as if it were a file
        byte = file.read(1)
        while byte:
            coded_file_ += huffman_codes_of_huffman_codes[byte]
            byte = file.read(1)

        compressed_huffman_codes, padding_ = Huffman.to_bytearray(coded_file_)   

        # Compute the length of the compressed file in bits
        compressed_file_length_ = len(compressed_huffman_codes) - padding_
        #End of encoding of Huffman code table with 1. order huffman alghorithm
        
        # Calculation of efficiency
        entropy = 0
        average_code_length = 0
        for key, p in probabilities.items():
            entropy += -p*np.log2(p) if p != 0 else 0
            average_code_length += p*len(huffman_codes[key])
        efficiency = entropy / average_code_length
        

        # Coding file with respect to the original huffman coding table
        coded_file = ''
        with open(filename_input, 'rb') as file:
            byte = file.read(tuple_size)
            while byte:
                # Padding with dummy NULL char  if needed
                if (len(byte) != tuple_size) & (len(byte) != 0):
                    byte += (N_padded_signs) * (b'\x00')
                coded_file += huffman_codes[byte]
                byte = file.read(tuple_size)
        
        compressed_data, padding = Huffman.to_bytearray(coded_file)    

        # Compute the length of the compressed file in bits
        compressed_file_length = len(compressed_data) - padding


        with open(filename_output, 'wb') as file:
            # Concatenate all data into one bytes object          
            file.write(b'CT:' + \
                            str(huffman_codes_of_huffman_codes).encode('utf-8') + \
                            b'\nNUM_BITS_CODE_TABLE:' + \
                            str(compressed_file_length_).encode('utf-8') + \
                            b'\nNUM_PADDED_BITS:' + \
                            str(padding_).encode('utf-8') + \
                            b'\nENCODED_CODE_TABLE:' + \
                            bitarray(compressed_huffman_codes) + \
                            b'\nNUMBER_OF_EFFECTIVE_BITS:' + \
                            str(compressed_file_length).encode('utf-8') + \
                            b'\nNUMBER_OF_PADED_SIGNS:' + \
                            str(N_padded_signs).encode('utf-8') + \
                            b'\nCODED_DATA:' + \
                            bitarray(compressed_data))

        return efficiency

    @staticmethod
    def decode_file(filename_input, filename_output):
        # rest of the decode_file code
        with open(filename_input, 'rb') as file:
            code_table_line = file.readline().decode('utf-8').strip()
            code_table_str = code_table_line.replace("CT:", "")
            huffman_encode_table = ast.literal_eval(code_table_str.strip())

            # Switch keys and values using a dictionary comprehension
            huffman_decode_table = {value: key for key, value in huffman_encode_table.items()}

            # Getting the number of bits of huffman encoded hufman table
            num_bits_line= file.readline().decode('utf-8').strip()
            num_bits = int(num_bits_line.split(':')[1].strip())

            # Getting the number of padded zeros at the end of huffman encoded hufman table
            num_padded_bits_line= file.readline().decode('utf-8').strip()
            num_padded_bits = int(num_padded_bits_line.split(':')[1].strip())

            file.read(19) # Reading the 'ENCODED_CODE_TABLE:'
            
            # Reading huffman table
            huffman_encoded_table = file.read(int((num_bits + num_padded_bits)/8))
            huffman_encoded_table_in_binarry = ''.join(format(b, '08b') for b in huffman_encoded_table)
            if num_padded_bits != 0:
                huffman_encoded_table_in_binarry = huffman_encoded_table_in_binarry[:-1*num_padded_bits]
            # Obtain huffman table from encoded one
            decoded_data = bytearray()
            checker = ""
            for i in range(num_bits):
                checker += huffman_encoded_table_in_binarry[i]
                if checker in huffman_decode_table:
                    # Check if we are at the end of a file, where padding kicks in
                    decoded_data.extend( huffman_decode_table[checker] )
                    checker = ""
            ####### FIX HERE .decode('utf-8').strip()
            byte_array_string = decoded_data.decode('utf-8').strip()
            huffman_decode_table = ast.literal_eval(byte_array_string.strip())

            # Switch keys and values using a dictionary comprehension
            huffman_decode_table = {value: key for key, value in huffman_decode_table.items()}

            file.read(26) # length of '\nNUMBER_OF_EFFECTIVE_BITS:'

            num_bits_actual_code= int(file.readline().decode('utf-8').strip())

            N_padded_signs= file.readline().decode('utf-8').strip()
            N_padded_signs = int(N_padded_signs.split(':')[1].strip())

            file.read(11) # len of 'CODED_DATA:'

            code = file.read()

            # Convert the input_bytes to a binary string
            encoded_data = ''.join(format(b, '08b') for b in code)
            decoded_data = bytearray()
            checker = ""
            for i in range(num_bits_actual_code):
                checker += encoded_data[i]
                if checker in huffman_decode_table:
                    # Check if we are at the end of a file, where padding kicks in
                    if (i == num_bits_actual_code - 1) & (N_padded_signs != 0):
                        decoded_data.extend( huffman_decode_table[checker][0:-1*N_padded_signs] )
                    else:
                        decoded_data.extend( huffman_decode_table[checker] )
                    checker = ""

        with open(filename_output, 'wb') as binary_file:
            binary_file.write(decoded_data)
                

def main():
    ###################################################################
    # NALOGA 1.
    print("####### NALOGA 1 #########\n")
    symbols_prob = {
        's1': 0.25,
        's2': 0.20,
        's3': 0.15,
        's4': 0.10,
        's5': 0.08,
        's6': 0.07,
        's7': 0.06,
        's8': 0.05,
        's9': 0.04,
    }
    huff = Huffman()
    huffman_codes = huff.huffman_algorithm(symbols_prob)
    print(huffman_codes)
    #################################################################
    # NALOGA 2., 3.
    print("\n\n ####### NALOGI 2 in 3 #########")


    file_names = ['datoteke/besedilo.txt','datoteke/besedilo.zip',
                  'datoteke/kodim01.bmp','datoteke/kodim02.bmp',
                  'datoteke/kodim03.bmp','datoteke/kodim04.bmp',
                  'datoteke/kodim05.bmp','datoteke/cartoon.bmp','datoteke/kodim01.png',
                  'datoteke/govor.wav','datoteke/narava.wav','datoteke/publika.wav','datoteke/govor.mp3']


    for file_name in file_names:
        filename_input_encoding = file_name
        print(f"\nDatoteka:{file_name}")
        for t in [1,2,3]:
            print(f"\nTerice dolzine {t}:\n")
            filename_output_encoding = "datoteke/generated_datoteke/" + file_name[9:-4] + "_encoded" + "_" + os.path.splitext(file_name)[1][1:]+ "_" +f"{t}" + ".huf"
            filename_output_decoding = "datoteke/generated_datoteke/" + file_name[9:-4] + "_decoded" + "_" + os.path.splitext(file_name)[1][1:] +"_" + f"{t}" + os.path.splitext(file_name)[1]
            filename_input_decoding = filename_output_encoding
            efficiency = Huffman.encode_file(filename_input_encoding, filename_output_encoding, tuple_size = t)
            print(f"     Uspesnost koda: {efficiency*100:.2f}%")
            # Doing the decoding 
            Huffman.decode_file(filename_input_decoding, filename_output_decoding)
            # Checking file sizes 
            file_size_original = os.path.getsize(file_name)   # Size of original file
            file_size_encoded = os.path.getsize(filename_output_encoding)   # Size of encoded file
            print(f"     Razmerje med velikostjo gospodarno kodirane in izvorne datoteke: {file_size_encoded/file_size_original:.2f}")

            with open(file_name,'rb') as f:
                data = f.read()
                original_md5 = md5_hash(data)
            with open(filename_output_decoding,'rb') as f:
                data = data = f.read()
                decompressed_md5 = md5_hash(data)
            if (original_md5 != decompressed_md5 ):
                print("NAPAKA, IZVLEČKA STA RAZLIČNA")

    ######################################################################
    # Testiranje kako velikost datoteke vpliva na kompresijsko razmerje

    # Dodatek za spreminjanje velikosti podatkov
    fractions = [0.2, 0.4, 0.6, 0.8, 1.0]  # Določitev želene frakcije dolžine datoteke

    with open('datoteke/besedilo.txt', 'rb') as f:
        data_full = f.read()  # Branje celotne datoteke

    full_size = len(data_full)  # Dolžina celotne datoteke

    # Slovarji za shranjevanje učinkovitosti in razmerja stiskanja
    efficiencies = {1: [], 2: [], 3: [], 4: []}
    compression_ratios = {1: [], 2: [], 3: [], 4: []}

    for fraction in fractions:
        data_size = int(full_size * fraction)  # Izračun dolžine podatkov
        data = data_full[:data_size]  # Izbor ustrezne količine podatkov

        # Tu zapišite podatke v začasno datoteko
        with open('temp.txt', 'wb') as f:
            f.write(data)

        for t in [1, 2, 3]:
            filename_input_encoding = 'temp.txt'
            filename_output_encoding = "datoteke/generated_datoteke/" + "temp" + "_encoded" + "_" + str(t) + ".huf"
            filename_output_decoding = "datoteke/generated_datoteke/" + "temp" + "_decoded" + "_" + str(t) + ".txt"
            filename_input_decoding = filename_output_encoding

            # Uporabite začasno datoteko namesto prvotne datoteke
            efficiency = Huffman.encode_file(filename_input_encoding, filename_output_encoding, tuple_size=t)
            Huffman.decode_file(filename_input_decoding, filename_output_decoding)

            # Preverjanje velikosti datotek 
            file_size_original = os.path.getsize('temp.txt')   # Velikost originalne datoteke
            file_size_encoded = os.path.getsize(filename_output_encoding)   # Velikost kodirane datoteke
            compression_ratio = file_size_encoded/file_size_original

            # Shranjevanje učinkovitosti in razmerja stiskanja
            efficiencies[t].append(efficiency)
            compression_ratios[t].append(compression_ratio)
        os.remove('temp.txt')


    # Plotting the data
    for t in [1, 2, 3]:
        plt.figure(figsize=(10, 5))
        plt.plot(fractions, compression_ratios[t], '-o')
        plt.title(f'Compression Ratio for Tuple Size {t}')
        plt.xlabel('Fraction of File Size')
        plt.ylabel('Compression Ratio')

        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    main()