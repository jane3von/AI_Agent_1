# AI_Agent_1
this is one AI Agent used Qwen API and Chroma DB to do the data analysis, all code from autogpt folder.

1. Use pip install all python relied lib in autogpt/requirements.txt
2. Configure your Qwen API Key (DASHSCOPE_API_KEY) in your environment. Use export DASHSCOPE_API_KEY="XXXyour true api key from QwenXXXX" in Mac system.
3. Download model all-MiniLM-L6-v2 to your loacal, it used in embedding_model variable in code autogpt/Utils/DocumentPipeline.py file.
   execute autogpt/loadModel.py to download.
5. Better to use conda env to run cmd: python main1.py
6. When first run main1.py, should run the code : pipeline.run() to create chroma db and load all data files to the db.
