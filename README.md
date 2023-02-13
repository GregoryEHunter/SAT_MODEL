# SAT_MODEL
 
## Data Downloading / generation / preprocessing
Data can be found in the data folder. Most preprocessing is done before training in the the "PaperSatTrainingAndGeneration folder" and in the "Baselines folder" For automatic evaluation the preprocessing is done in AutomaticEvaluation in the "Evaluation folder". The books preambles and other superfluous information was removed manually. First the emeddings must be generated first and this is done with the notebooks in the "Premake_embeddings" Folder addtionally the embeddings for Gatsby and Origins are made before training that model and which can be found in the "PaperSatTrainingAndGeneration" folder. All notebooks should be in the root directory as folder organization was done for ease of understanding. 
## training your baselines
Baseline training and extraction of excerpts can be found in the "Baslines" folder. This includes training all the finetuned GPT-2 models and extracting real excerpts from the datasets.
## training your experiments
Training of the SAT model variant used in this paper and generation for evaluation can be found in the "PaperSatTrainingAndGeneration" folder. These model definitions can be found in the "Model_definitions" folder, along with other model iterations. The correct model_import must be used and is found at the top of the jupiter notebook. In most cases "import Model_import_6" should be changed to "import Model_Import".
## Evaluating your model output (scoring, sampling, etc)
Automatic evaluation (including generation from the finetuned models and GPT-2 standard) can be found in the "Evaluation" folder. In the "Baselines" folder extraction of the real excerpts from the data can be found. Human evaluation used the same passages as the  Automaatic evaluation and the results from the human evaluation can be found in the Evaluation folder as well.
## OtherFinetuning has other finetuning techniques attempted and Misc contains other expierments that were done during the development process
