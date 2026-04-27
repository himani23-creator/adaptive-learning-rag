import streamlit as st
import os, json, re
from typing import TypedDict, List, Annotated
from datetime import datetime
import random # Added for random.choice

# LangChain components for building RAG and tool chains
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.tools import tool
from langchain_core.prompts import ChatPromptTemplate

# LangGraph is no longer used for direct execution flow in Streamlit, but its concepts are maintained
# from langgraph.graph import StateGraph, END
# from langgraph.prebuilt import ToolNode

# Groq LLM — ultra-fast inference with LLaMA models
from langchain_groq import ChatGroq

import wikipediaapi

import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Adaptive Learning Assistant", layout="wide")

st.title("🧠 Adaptive Learning Assistant")
st.write("Welcome to your personalized learning journey! I will teach you a topic, quiz you, and adapt to your progress.")

# --- Initialize LLM and Embedding Model and ChromaDB (Cached) ---
@st.cache_resource
def load_models():
    # Initialize LLM
    llm = ChatGroq(
        model="llama-3.3-70b-versatile", # Or any other suitable model
        temperature=0.3,
        max_tokens=1024,
        groq_api_key=st.secrets["GROQ_API_KEY"]
    )

    # Initialize embedding model
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Define curriculum
    curriculum = {
        "Python (programming language)":      ("Python", "beginner"),
        "Function (computer programming)":    ("Python", "intermediate"),
        "Supervised learning":                ("Machine Learning", "intermediate"),
        "Neural network (machine learning)":  ("Machine Learning", "advanced"),
        "Backpropagation":                    ("Machine Learning", "advanced"),
        "Linked list":                        ("Data Structures", "beginner"),
        "Binary search tree":                 ("Data Structures", "intermediate"),
        "Graph (abstract data type)":         ("Data Structures", "advanced"),
    }

    # Load raw documents from Wikipedia
    wiki = wikipediaapi.Wikipedia(
        language='en',
        user_agent='AdaptiveLearningBot/1.0'
    )

    raw_documents = []
    for topic_name, (subject, difficulty) in curriculum.items():
        page = wiki.page(topic_name)
        if page.exists():
            content = page.summary[:1500] # Limit content to 1500 chars
            raw_documents.append({
                "subject": subject,
                "topic": topic_name,
                "difficulty": difficulty,
                "content": content,
                "source_url": page.fullurl
            })

    # Convert to LangChain Documents
    documents = []
    for raw in raw_documents:
        doc = Document(
            page_content=raw["content"],
            metadata={
                "subject": raw["subject"],
                "topic": raw["topic"],
                "difficulty": raw["difficulty"],
                "source": raw["subject"] + "_" + raw["topic"].replace(" ", "_"),
                "source_url": raw.get("source_url", "")
            }
        )
        documents.append(doc)

    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", ". ", " "]
    )
    chunked_docs = text_splitter.split_documents(documents)

    # Initialize ChromaDB vectorstore
    CHROMA_PATH = "./chroma_edtech_db"
    vectorstore = Chroma.from_documents(
        documents=chunked_docs,
        embedding=embedding_model,
        persist_directory=CHROMA_PATH,
        collection_name="edtech_knowledge_base"
    )
    vectorstore.persist()

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    )

    return llm, retriever, vectorstore, curriculum

# Check for Groq API key in secrets
if "GROQ_API_KEY" not in st.secrets:
    st.error("GROQ_API_KEY not found in Streamlit secrets. Please add it to run the app.")
    st.stop()

llm, retriever, vectorstore, curriculum = load_models()

# --- Knowledge Base (Static) --- (This remains global as it's separate from data loading)
knowledge_base = {
    "quiz_questions": {
        "beginner": [
            {
                "question": "What type of programming language is Python and what is it mainly used for?",
                "answer": "Python is a high-level, interpreted, general-purpose language used for web development, data science, automation, and AI.",
                "topic": "Python (programming language)",
                "explanation": "Python's simple syntax and vast libraries make it one of the most popular languages worldwide."
            },
            {
                "question": "What is a linked list and how does it differ from an array?",
                "answer": "A linked list is a linear data structure where elements are stored in nodes connected by pointers, unlike arrays which store elements in contiguous memory.",
                "topic": "Linked list",
                "explanation": "Linked lists allow efficient insertions/deletions (O(1)) but have slower access (O(n)) compared to arrays."
            }
        ],
        "intermediate": [
            {
                "question": "What is a function in programming and why is it important?",
                "answer": "A function is a reusable block of code that performs a specific task, accepts inputs (parameters), and returns an output. It promotes code reusability and modularity.",
                "topic": "Function (computer programming)",
                "explanation": "Functions reduce code repetition, improve readability, and are the building blocks of structured programming."
            },
            {
                "question": "What is supervised learning and how does it differ from unsupervised learning?",
                "answer": "Supervise learning trains a model on labeled input-output pairs to predict outputs for new inputs. Unsupervised learning finds hidden patterns in unlabeled data.",
                "topic": "Supervise learning",
                "explanation": "In supervised learning the model learns from examples with known answers — like a student learning from a textbook with an answer key."
            },
            {
                "question": "What is a binary search tree and what is its key property?",
                "answer": "A BST is a tree where each node's left subtree contains only values less than the node, and the right subtree contains only values greater, enabling O(log n) search.",
                "topic": "Binary search tree",
                "explanation": "The ordering property of BSTs allows efficient search, insertion, and deletion compared to linear data structures."
            }
        ],
        "advanced": [
            {
                "question": "How does a neural network learn? Describe the role of weights and activation functions.",
                "answer": "Neural networks learn by adjusting weights through backpropagation to minimize loss. Activation functions like ReLU introduce non-linearity, allowing the network to model complex patterns.",
                "topic": "Neural network (machine learning)",
                "explanation": "Without activation functions, a neural network would just be a linear model regardless of depth. Non-linearity is what gives deep networks their expressive power."
            },
            {
                "question": "Explain backpropagation and the vanishing gradient problem.",
                "answer": "Backpropagation computes gradients of the loss function layer by layer using the chain rule. The vanishing gradient problem occurs when gradients become extremely small in early layers, stalling learning.",
                "topic": "Backpropagation",
                "explanation": "Solutions include ReLU activation, batch normalization, residual connections (ResNets), and careful weight initialization."
            },
            {
                "question": "What is a graph data structure and what are common algorithms used on graphs?",
                "answer": "A graph is a set of vertices connected by edges, used to model networks. Common algorithms include BFS, DFS, Dijkstra's shortest path, and Kruskal's minimum spanning tree.",
                "topic": "Graph (abstract data type)",
                "explanation": "Graphs power real-world systems like Google Maps (shortest path), social networks (friend suggestions), and compilers (dependency resolution)."
            }
        ]
    },
    "difficulty_progression": {
        "beginner": "intermediate",
        "intermediate": "advanced",
        "advanced": "advanced"
    }
}


# --- Helper function for filtered retrieval ---
def retrieve_with_filter(query: str, difficulty: str = None, subject: str = None):
    filter_conditions = []
    if difficulty:
        filter_conditions.append({"difficulty": difficulty})
    if subject:
        filter_conditions.append({"subject": subject})

    if filter_conditions:
        if len(filter_conditions) == 1:
            chroma_filter = filter_conditions[0]
        else:
            chroma_filter = {"$and": filter_conditions}
    else:
        chroma_filter = None

    filtered_retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": 3,
            "filter": chroma_filter
        }
    )
    return filtered_retriever.invoke(query)


# --- Tools ---
@tool
def retrieve_concept(query: str) -> str:
    """Retrieve educational content from the knowledge base for a given concept or topic.
    Use this to get relevant explanations before teaching a student."""

    docs = retriever.invoke(query)

    if not docs:
        return "No relevant content found for this topic."

    context = "\n\n".join([
        f"[{doc.metadata.get('topic', 'General')} | {doc.metadata.get('difficulty', '?').upper()}]\n{doc.page_content}"
        for doc in docs
    ])
    return context

@tool
def generate_quiz(topic: str, difficulty: str) -> str:
    """Generate a quiz question for the student based on a topic and difficulty level."""
    if difficulty not in knowledge_base["quiz_questions"]:
        return json.dumps({"question": "No questions available for this difficulty.", "answer": ""})

    available_questions = knowledge_base["quiz_questions"][difficulty]

    topic_questions = [q for q in available_questions if q.get("topic") == topic]

    if topic_questions:
        selected_question = random.choice(topic_questions)
    elif available_questions: # Fallback to any question of the difficulty if no topic match
        selected_question = random.choice(available_questions)
    else:
        return json.dumps({"question": "No questions available for this difficulty.", "answer": ""})

    return json.dumps({
        "question": selected_question["question"],
        "answer": selected_question["answer"],
        "explanation": selected_question["explanation"]
    })

@tool
def evaluate_answer(student_answer: str, expected_answer: str, topic: str) -> str:
    """Evaluate a student's answer against the expected answer using LLM-based semantic comparison.
    Returns pass/fail with detailed feedback."""

    if student_answer.strip().lower() == expected_answer.strip().lower():
        return json.dumps({"passed": True, "score": 100, "feedback": "Excellent! Your answer is spot on."})

    eval_prompt = f"""You are an answer evaluator. Compare the student's answer to the expected answer for the topic '{topic}'.
    Focus on semantic correctness and completeness.\n\nTopic: {topic}\nExpected Answer: {expected_answer}\nStudent Answer: {student_answer}\n\nYou MUST respond with ONLY a JSON object, no explanation, no markdown, no backticks.\nExample format: {{"passed": true, "score": 85, "feedback": "Good understanding shown."}}\n\nRespond now with JSON only:"""

    response = llm.invoke(eval_prompt)
    raw = response.content.strip()
    raw = re.sub(r"```json|```", "", raw).strip()

    json_match = re.search(r'\{.*?\}', raw, re.DOTALL)
    if json_match:
        try:
            parsed = json.loads(json_match.group())
            result = {
                "passed": bool(parsed.get("passed", False)),
                "score": int(parsed.get("score", 50)),
                "feedback": str(parsed.get("feedback", "Answer evaluated."))
            }
            return json.dumps(result)
        except json.JSONDecodeError:
            pass

    raw_lower = raw.lower()
    passed = any(word in raw_lower for word in ["correct", "good", "right", "yes", "true"])
    return json.dumps({
        "passed": passed,
        "score": 70 if passed else 40,
        "feedback": raw[:200] if raw else "Could not parse evaluation."
    })


# --- Define the shared state schema for direct agent interaction ---
class StudentState(TypedDict):
    student_name: str
    difficulty_level: str
    subject: str
    current_topic: str
    conversation_history: List[dict]
    quiz_question: str
    expected_answer: str
    student_answer: str
    evaluation_result: dict
    weak_areas: List[str]
    correct_count: int
    total_questions: int
    final_report: str
    # next_action: str # Removed as Streamlit will manage phase directly

# --- Agent 1: Tutor Agent ---
def tutor_agent_logic(state: StudentState) -> StudentState:
    topic = state["current_topic"]
    difficulty = state["difficulty_level"]
    subject = state["subject"]

    relevant_docs = retrieve_with_filter(topic, difficulty=difficulty, subject=subject)
    relevant_content = "\n\n".join([
        f"[{doc.metadata.get('topic', 'General')} | {doc.metadata.get('difficulty', '?').upper()}]\n{doc.page_content}"
        for doc in relevant_docs
    ])

    if not relevant_content:
        relevant_content = f"No specific content found for {topic} at {difficulty} level within {subject}. Providing a general explanation."
        general_docs = retriever.invoke(topic)
        relevant_content = "\n\n".join([
            f"[{doc.metadata.get('topic', 'General')} | {doc.metadata.get('difficulty', '?').upper()}]\n{doc.page_content}"
            for doc in general_docs
        ])
        if not general_docs:
            relevant_content = "No content found at all."

    prompt = f"""You are an adaptive AI tutor. Teach the following concept to a {difficulty} level student.\n\nRetrieved Knowledge:\n{relevant_content}\n\nStudent: {state['student_name']} | Level: {difficulty} | Topic: {topic}\n\nInstructions:\n- Explain clearly for a {difficulty} level student\n- Use simple examples if beginner, technical depth if advanced\n- End with: \"Ready for a quick quiz? (yes/no)\"\n\nProvide your explanation:"""

    response = llm.invoke(prompt)

    history = state["conversation_history"].copy()
    history.append({"role": "tutor", "content": response.content, "topic": topic})

    return {**state, "conversation_history": history}

# --- Agent 2: Quiz Agent ---
def quiz_agent_logic(state: StudentState) -> StudentState:
    topic = state["current_topic"]
    difficulty = state["difficulty_level"]

    quiz_json = generate_quiz.invoke({"topic": topic, "difficulty": difficulty})
    quiz_data = json.loads(quiz_json)

    return {
        **state,
        "quiz_question": quiz_data["question"],
        "expected_answer": quiz_data["answer"],
    }

# --- Agent 3: Progress Agent ---
def progress_agent_logic(state: StudentState) -> StudentState:
    topic = state["current_topic"]
    difficulty = state["difficulty_level"]

    eval_json = evaluate_answer.invoke({
        "student_answer": state["student_answer"],
        "expected_answer": state["expected_answer"],
        "topic": topic
    })

    try:
        eval_result = json.loads(eval_json) if eval_json else {}
        eval_result.setdefault("passed", False)
        eval_result.setdefault("score", 50)
        eval_result.setdefault("feedback", "No feedback available.")
    except (json.JSONDecodeError, TypeError):
        eval_result = {"passed": False, "score": 50, "feedback": "Evaluation parsing failed."}

    total = state["total_questions"] + 1
    correct = state["correct_count"] + (1 if eval_result.get("passed") else 0)
    weak_areas = state["weak_areas"].copy()

    if not eval_result.get("passed") or eval_result.get("score", 0) < 70:
        if topic not in weak_areas:
            weak_areas.append(topic)

    score = eval_result.get("score", 0)
    new_difficulty = state["difficulty_level"]

    # Determine next learning phase based on score
    next_phase_decision = ""
    if score >= 80:
        next_level = knowledge_base["difficulty_progression"][state["difficulty_level"]]
        new_difficulty = next_level
        next_phase_decision = "advance"
    elif score >= 50:
        next_phase_decision = "retry"
    else:
        next_phase_decision = "remediate"

    # Check if session limit is reached (for demo purposes)
    if total >= 3:
        next_phase_decision = "summary"

    return {
        **state,
        "evaluation_result": eval_result,
        "weak_areas": weak_areas,
        "total_questions": total,
        "correct_count": correct,
        "difficulty_level": new_difficulty,
        "next_action_decision": next_phase_decision # Store decision for Streamlit to use
    }

# --- Streamlit UI Logic ---

# Initialize Streamlit session state variables
if "langgraph_state" not in st.session_state:
    st.session_state.langgraph_state = None
if "current_phase" not in st.session_state:
    st.session_state.current_phase = "setup" # setup -> teaching -> quizzing -> evaluating -> summary_display
if "student_name_input_value" not in st.session_state:
    st.session_state.student_name_input_value = ""
if "subject_select_value" not in st.session_state:
    st.session_state.subject_select_value = "Python"
if "difficulty_select_value" not in st.session_state:
    st.session_state.difficulty_select_value = "beginner"
if "topic_select_value" not in st.session_state:
    # Initialize with a valid topic key from the curriculum
    st.session_state.topic_select_value = list(curriculum.keys())[0]

# Sidebar for user inputs
with st.sidebar:
    st.header("Your Learning Profile")
    st.session_state.student_name_input_value = st.text_input(
        "Your Name:", value=st.session_state.student_name_input_value,
        key="student_name_input"
    ).strip()

    st.session_state.subject_select_value = st.selectbox(
        "Subject:",
        options=list(set(item[0] for item in curriculum.values())),
        index=list(set(item[0] for item in curriculum.values())).index(st.session_state.subject_select_value),
        key="subject_select"
    )

    # Filter topics based on selected subject
    available_topics_for_subject_raw = [topic_name for topic_name, (sub, _) in curriculum.items() if sub == st.session_state.subject_select_value]
    if not available_topics_for_subject_raw:
        available_topics_for_subject = [list(curriculum.keys())[0]] # Fallback to a generic topic if no topics for subject
    else:
        # Sort topics alphabetically
        available_topics_for_subject = sorted(available_topics_for_subject_raw)

    st.session_state.difficulty_select_value = st.selectbox(
        "Your Level:",
        options=["beginner", "intermediate", "advanced"],
        index=["beginner", "intermediate", "advanced"].index(st.session_state.difficulty_select_value),
        key="difficulty_select"
    )

    # Determine the default topic for the topic selectbox based on subject AND difficulty
    # If the current topic_select_value is no longer valid for the selected subject, or if it's the very first session, try to find a better default.
    if st.session_state.topic_select_value not in available_topics_for_subject:
        found_topic = None
        for topic_name, (sub, diff) in curriculum.items():
            if sub == st.session_state.subject_select_value and diff == st.session_state.difficulty_select_value:
                found_topic = topic_name
                break

        if found_topic:
            st.session_state.topic_select_value = found_topic
        elif available_topics_for_subject: # Fallback to the first available topic for the subject if no difficulty match
            st.session_state.topic_select_value = available_topics_for_subject[0]

    # Ensure the topic_select_value is one of the options for the current subject before finding its index
    if st.session_state.topic_select_value not in available_topics_for_subject and available_topics_for_subject:
        st.session_state.topic_select_value = available_topics_for_subject[0]

    # Now set the index based on the (potentially updated) topic_select_value
    current_topic_idx = available_topics_for_subject.index(st.session_state.topic_select_value) if available_topics_for_subject else 0

    st.session_state.topic_select_value = st.selectbox(
        "Topic:",
        options=available_topics_for_subject,
        index=current_topic_idx,
        key="topic_select"
    )

    st.markdown("--- ")
    if st.session_state.langgraph_state:
        st.info(f"Current Phase: {st.session_state.current_phase.replace('_display', '').capitalize()}")
        st.info(f"Questions: {st.session_state.langgraph_state['correct_count']}/{st.session_state.langgraph_state['total_questions']}")
        st.info(f"Difficulty: {st.session_state.langgraph_state['difficulty_level'].capitalize()}")
    else:
        st.info("Current Phase: Setup")
    st.markdown("--- ")

    if st.session_state.current_phase == "summary_display" and st.session_state.langgraph_state and st.session_state.langgraph_state['final_report']:
        st.subheader("Session Summary")
        st.markdown(st.session_state.langgraph_state['final_report'])

# --- Callbacks for button actions ---
def start_session_callback():
    if not st.session_state.student_name_input_value:
        st.error("Please enter your name to start the session.")
        return

    initial_state = StudentState(
        student_name=st.session_state.student_name_input_value,
        difficulty_level=st.session_state.difficulty_select_value,
        subject=st.session_state.subject_select_value,
        current_topic=st.session_state.topic_select_value,
        conversation_history=[],
        quiz_question="",
        expected_answer="",
        student_answer="",
        evaluation_result={},
        weak_areas=[],
        correct_count=0,
        total_questions=0,
        final_report="",
    )
    st.session_state.langgraph_state = initial_state
    st.session_state.current_phase = "teaching" # Start teaching

def submit_answer_callback():
    student_answer = st.session_state.student_answer_input.strip()
    if not student_answer:
        st.error("Please type your answer before submitting.")
        return

    st.session_state.langgraph_state['student_answer'] = student_answer
    st.session_state.current_phase = "evaluating"

def proceed_to_quiz_callback():
    st.session_state.current_phase = "quizzing"

def continue_learning_callback():
    # Based on next_action_decision from progress_agent_logic
    decision = st.session_state.langgraph_state.get('next_action_decision')

    if decision == "advance" or decision == "remediate":
        st.session_state.current_phase = "teaching"
    elif decision == "retry":
        st.session_state.current_phase = "quizzing"
    elif decision == "summary":
        st.session_state.current_phase = "summary_display"
    else:
        st.error("Unexpected decision from progress agent. Restarting session.")
        st.session_state.current_phase = "setup"
        st.session_state.langgraph_state = None

# --- Main content area ---

if st.session_state.current_phase == "setup":
    st.button("Start Learning Session", on_click=start_session_callback, key="start_session_btn")

elif st.session_state.current_phase == "teaching":
    with st.spinner("Preparing your lesson..."):
        new_state = tutor_agent_logic(st.session_state.langgraph_state)
        st.session_state.langgraph_state = new_state

    explanation = new_state['conversation_history'][-1]['content'] if new_state['conversation_history'] else "No explanation available."
    st.subheader("Your Lesson")
    st.markdown(explanation)
    st.button("Ready for Quiz", on_click=proceed_to_quiz_callback, key="proceed_to_quiz_btn")

elif st.session_state.current_phase == "quizzing":
    with st.spinner("Generating quiz question..."):
        new_state = quiz_agent_logic(st.session_state.langgraph_state)
        st.session_state.langgraph_state = new_state

    st.subheader("Quiz Time!")
    st.write(new_state['quiz_question'])
    st.session_state.student_answer_input = st.text_area("Your Answer:", key="student_answer_ta")
    st.button("Submit Answer", on_click=submit_answer_callback, key="submit_answer_btn")

elif st.session_state.current_phase == "evaluating":
    with st.spinner("Evaluating your answer..."):
        new_state = progress_agent_logic(st.session_state.langgraph_state)
        st.session_state.langgraph_state = new_state

    eval_result = new_state['evaluation_result']
    st.subheader("Evaluation Result")
    if eval_result["passed"]:
        st.success(f"Score: {eval_result['score']}/100 - Passed!")
    else:
        st.error(f"Score: {eval_result['score']}/100 - Failed")
    st.write(f"**Feedback:** {eval_result['feedback']}")
    st.info(f"**Your Answer:** {new_state['student_answer']}")
    st.info(f"**Correct Answer:** {new_state['expected_answer']}")

    # Generate final report if decision is summary
    if new_state.get('next_action_decision') == "summary":
        accuracy = (new_state["correct_count"] / max(new_state["total_questions"], 1)) * 100
        report = f"""
        === LEARNING SESSION REPORT ===
        Student: {new_state['student_name']}
        Subject: {new_state['subject']} | Final Level: {new_state['difficulty_level']}

        Performance:
          - Questions Attempted: {new_state['total_questions']}
          - Correct Answers: {new_state['correct_count']}
          - Accuracy: {accuracy:.1f}%

        Weak Areas Identified: {', '.join(new_state['weak_areas']) if new_state['weak_areas'] else 'None — excellent performance!'}

        Recommendation: {'Advance to next subject' if accuracy >= 80 else 'Review weak areas before advancing'}
        Session completed: {datetime.now().strftime('%Y-%m-%d %H:%M')}
        """
        st.session_state.langgraph_state['final_report'] = report

    st.button("Continue", on_click=continue_learning_callback, key="continue_learning_btn")

elif st.session_state.current_phase == "summary_display":
    st.subheader("Session Complete!")
    st.markdown(st.session_state.langgraph_state['final_report'])
    if st.button("Start New Session", key="start_new_session_btn"):
        st.session_state.current_phase = "setup"
        st.session_state.langgraph_state = None
        st.rerun()

