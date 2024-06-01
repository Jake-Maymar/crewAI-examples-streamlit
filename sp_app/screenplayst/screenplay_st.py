import streamlit as st
from langchain_groq import ChatGroq
from crewai import Agent, Task, Crew, Process
import re
from io import StringIO
import sys

import langchain
langchain.verbose=True

class StreamToStreamlit:
    def __init__(self, streamlit_obj):
        self.streamlit_obj = streamlit_obj

    def write(self, text):
        self.streamlit_obj.markdown(text)

    def flush(self):
        pass

def main():
    with open("style.css") as css:
        st.markdown(f'<style>{css.read()}</style>', unsafe_allow_html=True)

    st.sidebar.title('Customization')
    model = st.sidebar.selectbox(
        'Choose a model',
        ['llama3-8b-8192', 'mixtral-8x7b-32768', 'gemma-7b-it', 'llama3-70b-8192']
    )
    llm = ChatGroq(
        temperature=0,
        groq_api_key=st.secrets["GROQ_API_KEY"],
        model_name=model
    )

    with st.sidebar:
        st.markdown("---")
        st.markdown(
            '<h5>Made with ❤ in &nbsp<img src="https://streamlit.io/images/brand/streamlit-mark-color.png" alt="Streamlit logo" height="16">&nbsp by <a href="https://twitter.com/jmaymar">@jmaymar</a></h5>',
            unsafe_allow_html=True,
        )

    st.title('Screenplay Writer')
    discussion = st.text_area("Paste any article or discussion here:")
    
    if st.button("Generate Screenplay"):
        spamfilter = Agent(
        role='spamfilter',
        goal='''Decide whether a text is spam or not.''',
        backstory='You are an expert spam filter with years of experience. You DETEST advertisements, newsletters and vulgar language.',
        llm=llm,
        verbose=True,
        allow_delegation=False
        )

        analyst = Agent(
            role='analyse',
            goal='''You will distill all arguments from all discussion members. Identify who said what. You can reword what they said as long as the main discussion points remain.''',
            backstory='You are an expert discussion analyst.',
            llm=llm,
            verbose=True,
            allow_delegation=False
        )

        scriptwriter = Agent(
            role='scriptwriter',
            goal='Turn a conversation into a movie script. Only write the dialogue parts. Do not start the sentence with an action. Do not specify situational descriptions. Do not write parentheticals.',
            backstory='''You are an expert on writing natural sounding movie script dialogues. You only focus on the text part and you HATE directional notes.''',
            llm=llm,
            verbose=True,
            allow_delegation=False
        )

        formatter = Agent(
            role='formatter',
            goal='''Format the text as asked. Leave out actions from discussion members that happen between brackets, eg (smiling).''',
            backstory='You are an expert text formatter.',
            llm=llm,
            verbose=True,
            allow_delegation=False
        )

        scorer = Agent(
        role='scorer',
        goal='''You score a dialogue assessing various aspects of the exchange between the participants using a 1-10 scale, where 1 is the lowest performance and 10 is the highest:
        Scale:
        1-3: Poor - The dialogue has significant issues that prevent effective communication.
        4-6: Average - The dialogue has some good points but also has notable weaknesses.
        7-9: Good - The dialogue is mostly effective with minor issues.
        10: Excellent - The dialogue is exemplary in achieving its purpose with no apparent issues.
        Factors to Consider:
        Clarity: How clear is the exchange? Are the statements and responses easy to understand?
        Relevance: Do the responses stay on topic and contribute to the conversation's purpose?
        Conciseness: Is the dialogue free of unnecessary information or redundancy?
        Politeness: Are the participants respectful and considerate in their interaction?
        Engagement: Do the participants seem interested and actively involved in the dialogue?
        Flow: Is there a natural progression of ideas and responses? Are there awkward pauses or interruptions?
        Coherence: Does the dialogue make logical sense as a whole?
        Responsiveness: Do the participants address each other's points adequately?
        Language Use: Is the grammar, vocabulary, and syntax appropriate for the context of the dialogue?
        Emotional Intelligence: Are the participants aware of and sensitive to the emotional tone of the dialogue?
        ''',
        backstory='You are an expert at scoring conversations on a scale of 1 to 10.',
        llm=llm,
        verbose=True,
        allow_delegation=False
        )

        # Filter out spam and vulgar posts
        task0 = Task(description='Read the following newsgroup post. If this contains vulgar language reply with STOP . If this is spam reply with STOP.\n### NEWGROUP POST:\n' + discussion,
        expected_output='Classification of the post as spam or containing vulgar language.',
        agent=spamfilter)
        result = task0.execute()
        if "STOP" in result:
            #stop here and proceed to next post
            print('This spam message will be filtered out')

        # process post with a crew of agents, ultimately delivering a well formatted dialogue
        task1 = Task(description='Analyse in much detail the following discussion:\n### DISCUSSION:\n' + discussion,
        expected_output='A detailed analysis of the discussion, identifying key points made by each participant.',
        agent=analyst)
        
        task2 = Task(description='Create a dialogue heavy screenplay from the discussion, between two persons. Do NOT write parentheticals. Leave out wrylies. You MUST SKIP directional notes.',
        expected_output='A screenplay dialogue based on the discussion, focusing on the conversation between two characters.',
        agent=scriptwriter)
        
        task3 = Task(description='''Format the script exactly like this:
            ## (person 1):
            (first text line from person 1)
                        
            ## (person 2):
            (first text line from person 2)
                        
            ## (person 1):
            (second text line from person 1)
                        
            ## (person 2):
            (second text line from person 2)
            
            ''',
            expected_output='The screenplay dialogue formatted according to the specified structure.',
            agent=formatter)

        with st.spinner("Running Workflow..."):
            crew = Crew(
                agents=[analyst, scriptwriter, formatter],
                tasks=[task1, task2, task3],
                verbose=2,
                process=Process.sequential,
                full_output=True,
            )
            # original_stdout = sys.stdout
            # sys.stdout = StreamToStreamlit(st)
            
            result = crew.kickoff()
            task3_export = task3._export_output
            task3_output = task3.output.raw_output
            task2_output = task2.output.raw_output
            # task1_output = task1.output.raw_output
            # print("task3XXXXXXXXXXXXXXXXXXXXX",result)
            # print("task3XXXXXXXXXXXXXXXXXXXXX",task3_export)
            # print("task3XXXXXXXXXXXXXXXXXXXXX",task3_output)
            # print("task2XXXXXXXXXXXXXXXXXXXXX",task2_output)
            # print("task1XXXXXXXXXXXXXXXXXXXXX",task1_output)
            # sys.stdout = original_stdout

            # Display the output of task3 in a new Streamlit text box
            st.subheader("Generated Screenplay:")
            # st.text_area("Task 3 Output", value=task3_output, height=400)
            # st.text_area("Task 2 Output", value=task2_output, height=400)
            # st.text_area("Task 1 Output", value=task1_output, height=400)
            if task3_output is None or task3_output.strip() == "":
                st.text_area("Task 2 Output", value=task2_output, height=400)
            else:
                st.text_area("Task 3 Output", value=task3_output, height=400)

            # st.text_area("Task 2 Output", value=task2_output, height=400)
            # st.text_area("Task 1 Output", value=task1_output, height=400)
    


if __name__ == "__main__":
    main()