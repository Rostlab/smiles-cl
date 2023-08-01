import os

# Disable tokenizers parallelism as this is not yet supported for custom tokenizers components
# Update: after refactoring the SMILES and byte tokenizers to use the native implementation,
#  this shouldn't be a problem anymore.
# os.environ["TOKENIZERS_PARALLELISM"] = "false"
