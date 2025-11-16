# Communication robot with AI agents

In this project, we focus on implementing a machine–human interface using state-of-the-art technologies 
such as LLMs, RAG, and AI agents. 

### Installing requirements
```
pip install -r requirements.txt
```

## Running Scripts
At the moment, the codebase is divided into several parts (this will change in the future). 
The repository also contains scripts from earlier phases of the thesis, which were used only to verify 
some of the underlying principles. Below is an overview of how to run the most important ones.

### Voice interaction
Wait until the ambient-noise threshold is set. After the terminal prompts you, you can ask your first question.  
Say **“stop”** to terminate the script.

### AI Agents
Separately from the voice interaction module, simple AI agents were implemented to control a “slider,” 
which acts as a hardware simulation.

Start the slider API in the first terminal. Once running, the slider interface will be available on the local host, 
where you can observe the values being set:
```
python3 eng_server_client/prepare_rag/populate_database.py
```
After database population you should have a `Chroma` folder with `chroma.splite3` file and vector data. 
In the first terminal, we run this script to start a simple MCP server—then we leave it running. 
```
python3 eng_server_client/minimalMCP.py
```
In the second terminal, run the agent client. In the console interface, you can instruct the agent to set an angle (SET) 
or retrieve the current angle (GET):
```
python3 eng_server_client/main_client.py
```
Wait until the threshold for ambient noise is set, and after the prompt in terminal you can ask the first question. 
You can stop the script by saying the word “stop.”

### AI agents
Independently of the voice interactions, simple AI agents were also implemented to plan the control of a “slider” 
(which serves as a hardware imitation). 

In the first terminal, we start the slider API. Once it loads, we can view the slider on the local host and monitor 
the values we set on it.
```
python3 ai_agents/slider_server.py
```
In the second terminal, we run the client. In the console interface, we can instruct the agent to set an angle (SET) 
or retrieve the current angle (GET).
```
python3 ai_agnets/planning_agent.py
```

### Tests
Testing scripts were prepared for experiments required during evaluation.  
To test the correctness of the RAG module, run:
```
python3 tests/test_RAG/main_test.py
```

[//]: # (```)

[//]: # (pip install langchain)

[//]: # (pip install -U langchain-community)

[//]: # (pip install sentence-transformers)

[//]: # (pip install hf_xet)

[//]: # (pip install chromadb)

[//]: # (pip install langchain-ollama)

[//]: # (```)
