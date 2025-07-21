# ui/app.py

import streamlit as st
import time # Used for simulating AI response time
import random # Used for randomizing mock responses

# --- Configuration for Mock Data ---
# These lists should ideally align with the data you used for fine-tuning
ROLES = ["AWS Cloud Engineer", "AI/ML Engineer", "Junior Python Developer"]
LEVELS = ["Junior", "Mid", "Senior"]
DOMAINS = {
    "AWS Cloud Engineer": ["Networking", "Compute", "Storage", "Security", "DevOps"],
    "AI/ML Engineer": ["Machine Learning Fundamentals", "Deep Learning", "MLOps", "NLP", "Computer Vision"],
    "Junior Python Developer": ["Data Structures & Algorithms", "Python Fundamentals & OOP"]
}
QUESTION_TYPES = ["Conceptual", "Scenario-based", "Problem Solving", "Coding"]


# --- Mock LLM Responses (Will be replaced by actual LLM inference later) ---
# This function simulates the LLM's behavior.
def mock_llm_response(prompt_type, role, level, domain, topic, q_type, question=None):
    st.session_state['thinking'] = True
    time.sleep(random.uniform(1.5, 3.0)) # Simulate thinking time
    st.session_state['thinking'] = False

    if prompt_type == "question":
        mock_questions = {
            "AWS Cloud Engineer": {
                "Networking": "Explain the difference between a Security Group and a Network ACL in AWS VPC.",
                "Compute": "Describe the benefits and use cases of AWS Lambda.",
                "Storage": "How would you choose between S3, EBS, and EFS for different storage needs?",
                "Security": "What is AWS IAM and how do you implement the principle of least privilege?",
                "DevOps": "Design a CI/CD pipeline for a serverless application on AWS."
            },
            "AI/ML Engineer": {
                "Machine Learning Fundamentals": "Explain the bias-variance trade-off and its implications in model training.",
                "Deep Learning": "Describe the architecture and working principle of a Convolutional Neural Network (CNN).",
                "MLOps": "What are the key challenges in deploying and monitoring ML models in production?",
                "NLP": "How do Transformer models address the limitations of Recurrent Neural Networks (RNNs) in sequence processing?",
                "Computer Vision": "Explain the concept of object detection and name a popular algorithm."
            },
            "Junior Python Developer": {
                "Data Structures & Algorithms": "Implement a function to reverse a singly linked list iteratively.",
                "Python Fundamentals & OOP": "Explain Python decorators and provide a simple example."
            }
        }
        return mock_questions.get(role, {}).get(domain, f"Mock question for {role} - {domain}.")
    elif prompt_type == "hint":
        return f"Hint: Consider the core components and their primary functions. Think about the OSI model layers."
    elif prompt_type == "explanation":
        return f"Explanation: A Security Group acts as a virtual firewall for instances, controlling traffic at the instance level. A Network ACL operates at the subnet level, providing stateless packet filtering. The key difference is statefulness and scope."
    elif prompt_type == "solution_code":
        return """
```python
# Mock Python Solution Code
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

def reverse_linked_list(head):
    prev = None
    current = head
    while current:
        next_node = current.next
        current.next = prev
        prev = current
        current = next_node
    return prev
```
"""
    return "Mock response."


# --- Streamlit UI Layout ---
st.set_page_config(layout="wide", page_title="InterviewReady AI Tutor", page_icon="🧠")

st.title("🧠 InterviewReady AI: Personalized Tech Interview Tutor")
st.markdown("""
Welcome to your AI-powered interview preparation assistant!
Select your desired role, level, domain, and question type to get started.
""")

# Initialize session state variables if they don't exist
if 'current_question' not in st.session_state:
    st.session_state.current_question = None
if 'current_hints' not in st.session_state:
    st.session_state.current_hints = []
if 'current_explanation' not in st.session_state:
    st.session_state.current_explanation = None
if 'current_solution_code' not in st.session_state:
    st.session_state.current_solution_code = None
if 'thinking' not in st.session_state:
    st.session_state.thinking = False

# Sidebar for user inputs
with st.sidebar:
    st.header("Configure Your Interview")

    selected_role = st.selectbox("Select Role", ROLES)
    selected_level = st.selectbox("Select Level", LEVELS)

    # Dynamically update domains based on selected role
    available_domains = DOMAINS.get(selected_role, [])
    selected_domain = st.selectbox("Select Domain", available_domains)

    selected_q_type = st.selectbox("Select Question Type", QUESTION_TYPES)
    selected_difficulty = st.select_slider("Select Difficulty", options=["Easy", "Medium", "Hard"])

    st.markdown("---")

    if st.button("Generate New Question", use_container_width=True, disabled=st.session_state.thinking):
        st.session_state.current_question = None
        st.session_state.current_hints = []
        st.session_state.current_explanation = None
        st.session_state.current_solution_code = None

        with st.spinner("Generating question..."):
            # Mock LLM call for question generation
            st.session_state.current_question = mock_llm_response(
                "question", selected_role, selected_level, selected_domain, "N/A", selected_q_type
            )
        st.success("Question generated!")

# Main content area
if st.session_state.current_question:
    st.subheader("Your Interview Question:")
    st.info(st.session_state.current_question)

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("Get Hint", use_container_width=True, disabled=st.session_state.thinking):
            with st.spinner("Getting hint..."):
                new_hint = mock_llm_response(
                    "hint", selected_role, selected_level, selected_domain, "N/A", selected_q_type,
                    question=st.session_state.current_question
                )
                st.session_state.current_hints.append(new_hint)
            st.success("Hint provided!")

    with col2:
        if st.button("Get Explanation", use_container_width=True, disabled=st.session_state.thinking):
            with st.spinner("Getting explanation..."):
                st.session_state.current_explanation = mock_llm_response(
                    "explanation", selected_role, selected_level, selected_domain, "N/A", selected_q_type,
                    question=st.session_state.current_question
                )
            st.success("Explanation provided!")

    with col3:
        if st.button("Get Solution Code", use_container_width=True, disabled=st.session_state.thinking):
            with st.spinner("Getting solution code..."):
                st.session_state.current_solution_code = mock_llm_response(
                    "solution_code", selected_role, selected_level, selected_domain, "N/A", selected_q_type,
                    question=st.session_state.current_question
                )
            st.success("Solution code provided!")

    # Display hints
    if st.session_state.current_hints:
        st.subheader("Hints:")
        for i, hint in enumerate(st.session_state.current_hints):
            st.warning(f"Hint {i+1}: {hint}")

    # Display explanation
    if st.session_state.current_explanation:
        st.subheader("Explanation:")
        st.success(st.session_state.current_explanation)

    # Display solution code
    if st.session_state.current_solution_code:
        st.subheader("Solution Code:")
        st.code(st.session_state.current_solution_code, language="python")

else:
    st.info("Click 'Generate New Question' in the sidebar to begin your interview prep!")

st.markdown("---")
st.caption("Powered by InterviewReady AI (LLM integration pending)")
