# AI-Business-Insight-assistant

This project provides a comprehensive analysis of sales data using Python and integrates an AI agent for interactive recommendations and insights. The code is divided into several sections, including data preprocessing, advanced data summarization, visualization, and an AI-powered agent for answering user queries.

---

## Table of Contents
1. [Introduction](#introduction)
2. [Code Overview](#code-overview)
3. [Data Analysis](#data-analysis)
4. [Visualization](#visualization)
5. [AI Agent](#ai-agent)
6. [Model Evaluation](#model-evaluation)
7. [Streamlit App](#streamlit-app)

---

## Introduction
This project analyzes sales data to extract key insights, such as total sales, best/worst performing months, and top-selling products. It also includes an AI agent powered by OpenAI's GPT-3.5-turbo model to provide recommendations and answer user queries based on the analyzed data.

---

## Code Overview
The code is structured as follows:
1. **Data Loading and Preprocessing**: Load the sales data and preprocess it for analysis.
2. **Advanced Data Summary**: Compute key metrics such as total sales, average sales, and best/worst performing months.
3. **Visualization**: Generate visualizations of sales trends by product.
4. **AI Agent**: Create an AI agent using LangChain and OpenAI to provide recommendations and answer questions.
5. **Model Evaluation**: Evaluate the AI agent's performance using predefined question-answer pairs.
6. **Streamlit App**: Build an interactive web app using Streamlit for data visualization and AI agent interaction.

---

## Data Analysis
### Key Metrics
- **Total Sales**: Sum of all sales.
- **Average Sales**: Mean sales value.
- **Standard Deviation of Sales**: Measure of sales variability.
- **Best/Worst Month**: Month with the highest/lowest sales.
- **Top-Selling Product**: Product with the highest total sales.
- **Most Frequent Product**: Product with the highest number of sales transactions.

## Visualization
The plt_sales_by_product function generates a plot of smoothed sales data for each product over time.

## Model Evaluation
The AI agent is evaluated using predefined question-answer pairs to ensure accuracy.

## AI Agent
The AI agent is built using LangChain and OpenAI's GPT-3.5-turbo model. It provides recommendations and answers user queries based on the sales data summary.

Tools
OverallChainTool: Provides recommendations based on the sales data summary.

GeneralChatTool: Handles unrelated enquiries.

RecallHistoryTool: Recalls chat history if requested.

```python


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



## Streamlit App
The Streamlit app provides an interactive interface for data visualization and AI agent interaction.

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
