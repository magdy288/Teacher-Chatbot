from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.graph.graph import CompiledGraph


MODEL = ChatOllama(
    model='llama3.2:3b',
    temperature=0,
)


def call_partygoer(state: MessagesState):
    system = SystemMessage(
        'Your name is Maria. ' 
        'You are currently at a party and you are having a great time. ' 
        'You are talkative and curious, and you love to learn more about people. ' 
        'Respond only with words, not actions.'
    )
    response = MODEL.invoke([system] + state['messages'][:-1])
    response.response_metadata['ai_name'] = 'partygoer'
    return {'messages': [response]}


def call_teacher(state: MessagesState):
    system = SystemMessage(
        'You are a English teacher at a social gathering, discreetly helping your' 
        'student to practice English while talking to others. While you ' 
        'student talks to others, you: \ n \ n'
        '- You focus only on essential corrections for natural English' 
        'spoken \ n' 
        '- You suggest more idiomatic alternatives or that sound more native when it is' 
        'relevant \ n' 
        '- You keep your advice brief and discreet, as a whispered suggestion \ n' 
        '- You only comment on the most recent statement of your student \ n' 
        '- You respond only with corrections or verbal suggestions \ n' 
        '- You say "I have no suggestions." if your English was natural and appropriate \ n \ n ' 
        'For example: \ n' 'Student: "I am going to the store yesterday." \ N ' 
        'You: "Fast correction: I went to the store yesterday" \ n \ n ' 
        'Student: "The weather is very good today!"\ n ' 
        'You: "I have no suggestions." \ N \ n ' 
        'Student: "This party gives me a lot of fun." \ N ' 
        'You: "More natural to say: This party is a lot of fun!"'
    )
    response = MODEL.invoke([system] + state['messages'])
    response.response_metadata['ai_name'] = 'teacher'
    return {'messages': [response]}


def build_graph() -> CompiledGraph:
    workflow = StateGraph(MessagesState)

    workflow.add_node('teacher', call_teacher)
    workflow.add_node('partygoer', call_partygoer)

    workflow.add_edge(START, 'teacher')
    workflow.add_conditional_edges(
        'teacher',
        lambda s: s['messages'][-1].content.startswith('No tengo sugerencias'),
        {
            True: 'partygoer',
            False: END,
        },
    )
    workflow.add_edge('partygoer', END)

    checkpointer = MemorySaver()

    return workflow.compile(checkpointer=checkpointer)