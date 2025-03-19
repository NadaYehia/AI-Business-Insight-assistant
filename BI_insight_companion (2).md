# AI-Business-Insight-assistant

This project provides a comprehensive analysis of sales data using Python and integrates an AI agent for interactive recommendations and insights. The code is divided into several sections, including data preprocessing, advanced data summarization, visualization, and an AI-powered agent for answering user queries.



```python
import pandas as pd
import numpy as np
import scipy 
import json
from langchain import PromptTemplate, LLMChain
from langchain.chains import SequentialChain
from langchain.evaluation import QAEvalChain
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
import random
from langchain.tools import Tool
from langchain.agents import load_tools, initialize_agent, AgentType
from langchain.memory import ConversationBufferMemory
from langchain.agents import ZeroShotAgent, AgentExecutor
import streamlit as st
from functools import partial

# Step 2: Compute the Advanced Data Summary

df=pd.read_csv('sales_data.csv')
df.head()
df.isna().sum()

# step 1
df['Date_dt']=pd.to_datetime(df['Date'])
df['Month']=df['Date_dt'].dt.month


total_sales=df['Sales'].sum()
average_sale=df['Sales'].mean()
standard_dev_sales=df['Sales'].std()

# step 3
sum_sales_by_month=np.array(df.groupby('Month')['Sales'].sum())
best_month=np.argmax(sum_sales_by_month)+1
worst_month=np.argmin(sum_sales_by_month)+1

# step 4 a
# top selling product (product with max sum sales)
sales_aggr_by_product=df.groupby('Product')['Sales'].sum()
bp_idx=np.argmax(sales_aggr_by_product)
bp=sales_aggr_by_product.keys()[bp_idx]

# most frequent product (product with more sales counts)
product_sales_freq=df['Product'].value_counts()
fqtp_idx=np.argmax(product_sales_freq)
fqtp=product_sales_freq.keys()[fqtp_idx]

advanced_data_summary= f" total sales for all products is {str(total_sales)} with mean { str(average_sale)} and standard deviation {str(standard_dev_sales)}"\
                     f"\n Month with best overall sales is {str(best_month)} and the worst is in {str(worst_month)}."\
                      f"\nFinally the top selling product is {str(bp)} and the most frequent one is {str(fqtp)}"



## Visualization
def plt_sales_by_product(df):
    
    df=df.reset_index()
    df.set_index("Date_dt",inplace=True)
    sales_by_product=df.groupby('Product')
    
    fig , axs=plt.subplots(1,len(sales_by_product),figsize=(30,10),sharey=True)
    viridis = mpl.colormaps['viridis'].resampled(8)
    
    k=0
    for grp_name,data_group in sales_by_product:
        smoothed_data=data_group["Sales"].rolling(window=50).mean()
        axs[k].plot(smoothed_data.index,smoothed_data,label=grp_name,color=viridis(random.random()),alpha=0.8)
        axs[k].set_xlabel("Year",fontsize=30)
        axs[k].set_ylabel("Sales",fontsize=30)
        print(np.linspace(np.min(smoothed_data),np.max(smoothed_data),5))
        axs[k].set_xticklabels(np.linspace(np.min((smoothed_data.index.year)),np.max((smoothed_data.index.year)),7),fontsize=20,rotation=45)
        axs[k].set_yticklabels(np.linspace(np.min(smoothed_data),np.max(smoothed_data),5), fontsize=20)
        axs[k].legend(loc='best',fontsize=20)
        k=k+1
    plt.show()
    return fig


## define first agent tool, sequential chain to provide analyses and recommendations 
llm = ChatOpenAI(model_name="gpt-3.5-turbo",temperature=0)

expert_AI_template="""
Analyze the following sales data summary {advanced_data_summary}

and answer {question} from {advanced_data_summary}. Finally, Provide a consice analysis of the key points:
"""

data_analysis_prompt=PromptTemplate(template=expert_AI_template,input_variables=["advanced_data_summary","question"])
data_analysis_chain=LLMChain(llm=llm, prompt=data_analysis_prompt,output_key="analysis")

recommendation_template= """
Based on the following analysis of sales data:
{analysis}
provide specific recommendations to address the question: {question}

"""
recommendation_prompt=PromptTemplate(template=recommendation_template,input_variables=["analysis","question"])
recommendation_chain=LLMChain(llm=llm,prompt=recommendation_prompt,output_key='recommendations')

overall_chain = SequentialChain(chains = [data_analysis_chain, recommendation_chain],
                            input_variables=['advanced_data_summary','question'],
                            output_variables=['analysis', 'recommendations'],
                            verbose=True
                           )

def sequential_chain_tool_func(question,advanced_data_summary):
    # Run the SequentialChain
    result = overall_chain({"advanced_data_summary":advanced_data_summary,
                            "question":question})
    return f"Data analysis: {result['analysis']}\n Recommendation: {result['recommendations']}"



# tool #2 and 3
# Tool 2: General Chat Tool
def general_chat_tool(question: str) -> str:
    """Handle unrelated enquiries."""
    return f"I'm sorry, I don't have information about '{question}'. How can I assist you otherwise?"

def recall_history_tool(question: str, chat_history: str) -> str:
    #print(question)
    """Recall chat history if the user asks."""
    return f"Chat History:\n{chat_history}"
    
tools = [
    Tool(
        name="OverallChainTool",
        func= partial(sequential_chain_tool_func, advanced_data_summary=advanced_data_summary),
        description="Based on advanced data summary and its analysis, provide specific recommendations to address question"
    ),
    Tool(
        name="GeneralChatTool",
        func=general_chat_tool,
        description="Use this tool to handle any enquiries unrelated to sales and business performance."
    ),
     Tool(
        name="RecallHistoryTool",
        func=lambda question: recall_history_tool(question, memory.buffer),
        description="Use this tool to recall chat history if the user asked for it. You can use this tool to recall information the user provided to you earlier."
    )
]

memory = ConversationBufferMemory(memory_key="chat_history")
prompt= ZeroShotAgent.create_prompt(
    tools=tools,
    prefix="You are a helpful AI assistant. Your goal is to assist the user by processing their requests using the tools available to you.",
    suffix="Begin!\n\nChat History:\n{chat_history}\n\nQuestion: {input}\nThought: {agent_scratchpad}"
)

llm_chain=LLMChain(llm=llm,prompt=prompt)
#Initialize the Agent
agent = ZeroShotAgent(llm_chain=llm_chain, tools=tools, verbose=True)

#Agent executor
agent_chain=AgentExecutor.from_agent_and_tools(agent=agent,tools=tools,memory=memory,verbose=True,handle_parsing_errors=True)

## Model evaluation
qa_pairs = [
        {
            "question": "What is our total sales amount?",
            "answer": f"The total sales amount is ${df['Sales'].sum():,.2f}."
        },
        {
            "question": "Which product category has the highest sales?",
            "answer": f"The product category with the highest sales is {df.groupby('Product')['Sales'].sum().idxmax()}."
        },
        {
            "question": "What is our average customer satisfaction score?",
            "answer": f"The average customer satisfaction score is {df['Customer_Satisfaction'].mean():.2f}."
        },
    ]    

def evaluate_model(qa_pairs,advanced_data_summary):
    eval_chain=QAEvalChain.from_llm(llm=llm)
    # predictions
    predictions=[]
    for qs in qa_pairs:
        result=agent_chain.run(qs["question"])
        predictions.append({"question":qs['question'], "prediction":result})
    
    eval_results=eval_chain.evaluate(examples=qa_pairs,
                        predictions=predictions,
                        prediction_key="prediction",
                        question_key="question",
                        answer_key="answer")
    mod_eval=[]
    for i,res in enumerate(eval_results):
        print(res)
        mod_eval.append({
            "question":qa_pairs[i]["question"],
            "actual":qa_pairs[i]["answer"],
            "predicted": predictions[i]["prediction"],
            "correct": res["results"]=='CORRECT'
            
        })
    return mod_eval
            
#model_eval=evaluate_model(qa_pairs,advanced_data_summary)

```

    /tmp/ipykernel_69/1185357666.py:85: LangChainDeprecationWarning: The class `ChatOpenAI` was deprecated in LangChain 0.0.10 and will be removed in 1.0. An updated version of the class exists in the :class:`~langchain-openai package and should be used instead. To use it run `pip install -U :class:`~langchain-openai` and import as `from :class:`~langchain_openai import ChatOpenAI``.
      llm = ChatOpenAI(model_name="gpt-3.5-turbo",temperature=0)
    /tmp/ipykernel_69/1185357666.py:94: LangChainDeprecationWarning: The class `LLMChain` was deprecated in LangChain 0.1.17 and will be removed in 1.0. Use :meth:`~RunnableSequence, e.g., `prompt | llm`` instead.
      data_analysis_chain=LLMChain(llm=llm, prompt=data_analysis_prompt,output_key="analysis")
    /tmp/ipykernel_69/1185357666.py:148: LangChainDeprecationWarning: Please see the migration guide at: https://python.langchain.com/docs/versions/migrating_memory/
      memory = ConversationBufferMemory(memory_key="chat_history")
    /tmp/ipykernel_69/1185357666.py:157: LangChainDeprecationWarning: LangChain agents will continue to be supported, but it is recommended for new use cases to be built with LangGraph. LangGraph offers a more flexible and full-featured framework for building agents, including support for tool-calling, persistence of state, and human-in-the-loop workflows. For details, refer to the `LangGraph documentation <https://langchain-ai.github.io/langgraph/>`_ as well as guides for `Migrating from AgentExecutor <https://python.langchain.com/docs/how_to/migrate_agent/>`_ and LangGraph's `Pre-built ReAct agent <https://langchain-ai.github.io/langgraph/how-tos/create-react-agent/>`_.
      agent = ZeroShotAgent(llm_chain=llm_chain, tools=tools, verbose=True)
    


```python
agent_chain.run("can you recall the chat history?")
```

    
    
    [1m> Entering new AgentExecutor chain...[0m
    [32;1m[1;3mThe user is asking to recall the chat history.
    Action: RecallHistoryTool
    Action Input: Recall chat history[0m
    Observation: [38;5;200m[1;3mChat History:
    Human: how to enhance our sales performance?
    AI: To enhance sales performance, the company should focus on increasing sales during the month of December, run promotions, analyze the success of top-selling products like Widget A, gather customer feedback, personalize marketing efforts, invest in staff training, utilize social media, and regularly monitor sales performance.
    Human: tell me a joke
    AI: Agent stopped due to iteration limit or time limit.[0m
    Thought:[32;1m[1;3mI have successfully recalled the chat history for the user.
    
    Final Answer: Chat History:
    Human: how to enhance our sales performance?
    AI: To enhance sales performance, the company should focus on increasing sales during the month of December, run promotions, analyze the success of top-selling products like Widget A, gather customer feedback, personalize marketing efforts, invest in staff training, utilize social media, and regularly monitor sales performance.
    Human: tell me a joke
    AI: Agent stopped due to iteration limit or time limit.[0m
    
    [1m> Finished chain.[0m
    




    'Chat History:\nHuman: how to enhance our sales performance?\nAI: To enhance sales performance, the company should focus on increasing sales during the month of December, run promotions, analyze the success of top-selling products like Widget A, gather customer feedback, personalize marketing efforts, invest in staff training, utilize social media, and regularly monitor sales performance.\nHuman: tell me a joke\nAI: Agent stopped due to iteration limit or time limit.'




```python


## Streamlit app code
# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Data Visualization", "AI agent"])
st.page_icon=":robot:"

# Display the selected page
if page == "Home":
    st.title("Welcome to your AI business companion!")
    st.write("This is the home page. Use the navigation panel on the left to explore other sections.")
    
elif page == "Data Visualization":
    st.title("Data Visualization and summary")
    st.write("Smoothed out sales Data for all our products")
    fig=plt_sales_by_product(df)
    st.pyplot(fig)
    st.write(f"Here is some key metrics summarized from our sales data\n {advanced_data_summary}")
    
elif page == "AI agent":
    
    st.title("AI agent")
    st.write("Ask me anything!")

    # User input
    user_input = st.text_input("Enter your question:")
    if user_input:
        
        # Generate chatbot response
        response = agent_chain.run(user_input)
        st.write("### Response:")
        st.write(response)

```
