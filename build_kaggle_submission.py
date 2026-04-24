import os
import base64
import zlib

def build_submission():
    pkl_file = "best_bot.pkl"
    py_file = "best_bot.py"
    
    out_file = "submission.py"

    if not os.path.exists(pkl_file):
        print(f"Error: {pkl_file} not found!")
        return

    with open(pkl_file, "rb") as f:
        weights_data = f.read()

    # Compress the weights
    print("Compressing weights...")
    compressed_data = zlib.compress(weights_data)
    encoded_weights = base64.b64encode(compressed_data).decode('utf-8')

    with open(py_file, "r") as f:
        bot_code = f.read()

    # Injected logic
    injection = f"""
import io
import zlib
import sys
import base64
import HTMRL.temporal_memory as temporal_memory

class CustomUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if name == 'SpatialPooler':
            return SpatialPooler
        if name == 'TemporalMemory':
            return temporal_memory.TemporalMemory
        return super().find_class(module, name)

WEIGHTS_B64 = '{encoded_weights}'
"""
    
    # Insert injection at the top after imports
    bot_code = bot_code.replace("import os", "import os" + injection)

    # Decompress before unpickling
    old_load_logic = """        if load_path and os.path.exists(load_path):\n            with open(load_path, "rb") as f:\n                self.sp = pickle.load(f)"""
                
    new_load_logic = """        if WEIGHTS_B64:\n            # Decompress and then unpickle using the custom unpickler\n            decompressed = zlib.decompress(base64.b64decode(WEIGHTS_B64))\n            self.sp = CustomUnpickler(io.BytesIO(decompressed)).load()"""

    bot_code = bot_code.replace(old_load_logic, new_load_logic)
    
    # Standardize entry point name to 'agent' with robust signature (obs, config=None)
    # This handles both 1-arg and 2-arg calls from the Kaggle environment
    bot_code = bot_code.replace("def agent_fn(obs):", "def agent(obs, config=None):")
    bot_code = bot_code.replace("def agent(obs):", "def agent(obs, config=None):")

    with open(out_file, "w") as f:
        f.write(bot_code)
    size_kb = os.path.getsize(out_file) / 1024
    print(f"Success! Created {out_file} ({size_kb:.2f} KB).")

    if os.path.getsize(out_file) > 1000 * 1024:
        print("WARNING: File exceeds 1MB limit!")

if __name__ == "__main__":
    build_submission()
