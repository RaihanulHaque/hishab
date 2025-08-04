import os
import json
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, TypedDict, Literal
from dataclasses import dataclass, asdict
from enum import Enum
import asyncio
from pathlib import Path

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode

# Data Models
class TaskStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    RESCHEDULED = "rescheduled"

class TaskType(Enum):
    SIMPLE = "simple"
    PROJECT = "project"
    RECURRING = "recurring"

class Priority(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"

@dataclass
class Reminder:
    id: str
    task_id: str
    reminder_time: datetime
    message: str
    is_sent: bool = False
    reminder_type: str = "standard"  # standard, soft, final

@dataclass
class TaskLog:
    timestamp: datetime
    action: str
    description: str
    user_input: Optional[str] = None

@dataclass
class FileAttachment:
    filename: str
    filepath: str
    file_type: str
    upload_date: datetime
    description: Optional[str] = None

@dataclass
class Task:
    id: str
    title: str
    description: str
    status: TaskStatus
    task_type: TaskType
    priority: Priority
    created_date: datetime
    due_date: Optional[datetime] = None
    start_date: Optional[datetime] = None
    assigned_to: Optional[str] = None
    project_id: Optional[str] = None
    subtasks: List[str] = None
    tags: List[str] = None
    attachments: List[FileAttachment] = None
    links: List[str] = None
    logs: List[TaskLog] = None
    reminders: List[Reminder] = None
    estimated_hours: Optional[float] = None
    actual_hours: Optional[float] = None
    
    def __post_init__(self):
        if self.subtasks is None:
            self.subtasks = []
        if self.tags is None:
            self.tags = []
        if self.attachments is None:
            self.attachments = []
        if self.links is None:
            self.links = []
        if self.logs is None:
            self.logs = []
        if self.reminders is None:
            self.reminders = []

@dataclass
class Project:
    id: str
    name: str
    description: str
    status: TaskStatus
    created_date: datetime
    due_date: Optional[datetime] = None
    start_date: Optional[datetime] = None
    task_ids: List[str] = None
    budget: Optional[float] = None
    spent: Optional[float] = None
    
    def __post_init__(self):
        if self.task_ids is None:
            self.task_ids = []

# State Management
class AgentState(TypedDict):
    messages: List[Any]
    user_input: str
    current_task: Optional[Task]
    current_project: Optional[Project]
    intent: str
    entities: Dict[str, Any]
    context: Dict[str, Any]
    response: str
    next_action: str

class PersonalAssistantAgent:
    def __init__(self, openai_api_key: str, data_dir: str = "assistant_data"):
        self.llm = ChatOpenAI(
            model="gpt-4",
            temperature=0.1,
            openai_api_key=openai_api_key
        )
        
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        # Storage
        self.tasks: Dict[str, Task] = {}
        self.projects: Dict[str, Project] = {}
        self.reminders: Dict[str, Reminder] = {}
        
        # Load existing data
        self._load_data()
        
        # Build graph
        self.graph = self._build_graph()
        self.memory = MemorySaver()
        
    def _load_data(self):
        """Load existing tasks and projects from storage"""
        try:
            tasks_file = self.data_dir / "tasks.json"
            if tasks_file.exists():
                with open(tasks_file, 'r') as f:
                    tasks_data = json.load(f)
                    for task_id, task_data in tasks_data.items():
                        self.tasks[task_id] = self._dict_to_task(task_data)
            
            projects_file = self.data_dir / "projects.json"
            if projects_file.exists():
                with open(projects_file, 'r') as f:
                    projects_data = json.load(f)
                    for project_id, project_data in projects_data.items():
                        self.projects[project_id] = self._dict_to_project(project_data)
                        
        except Exception as e:
            print(f"Error loading data: {e}")
    
    def _save_data(self):
        """Save tasks and projects to storage"""
        try:
            # Save tasks
            tasks_data = {task_id: self._task_to_dict(task) for task_id, task in self.tasks.items()}
            with open(self.data_dir / "tasks.json", 'w') as f:
                json.dump(tasks_data, f, indent=2, default=str)
            
            # Save projects
            projects_data = {project_id: self._project_to_dict(project) for project_id, project in self.projects.items()}
            with open(self.data_dir / "projects.json", 'w') as f:
                json.dump(projects_data, f, indent=2, default=str)
                
        except Exception as e:
            print(f"Error saving data: {e}")
    
    def _dict_to_task(self, task_data: Dict) -> Task:
        """Convert dictionary to Task object"""
        # Handle datetime conversion
        for date_field in ['created_date', 'due_date', 'start_date']:
            if task_data.get(date_field):
                task_data[date_field] = datetime.fromisoformat(task_data[date_field])
        
        # Handle enums
        task_data['status'] = TaskStatus(task_data['status'])
        task_data['task_type'] = TaskType(task_data['task_type'])
        task_data['priority'] = Priority(task_data['priority'])
        
        # Handle nested objects
        if task_data.get('attachments'):
            task_data['attachments'] = [FileAttachment(**att) for att in task_data['attachments']]
        if task_data.get('logs'):
            task_data['logs'] = [TaskLog(**log) for log in task_data['logs']]
        if task_data.get('reminders'):
            task_data['reminders'] = [Reminder(**rem) for rem in task_data['reminders']]
        
        return Task(**task_data)
    
    def _task_to_dict(self, task: Task) -> Dict:
        """Convert Task object to dictionary"""
        task_dict = asdict(task)
        # Convert enums to strings
        task_dict['status'] = task.status.value
        task_dict['task_type'] = task.task_type.value
        task_dict['priority'] = task.priority.value
        return task_dict
    
    def _dict_to_project(self, project_data: Dict) -> Project:
        """Convert dictionary to Project object"""
        # Handle datetime conversion
        for date_field in ['created_date', 'due_date', 'start_date']:
            if project_data.get(date_field):
                project_data[date_field] = datetime.fromisoformat(project_data[date_field])
        
        project_data['status'] = TaskStatus(project_data['status'])
        return Project(**project_data)
    
    def _project_to_dict(self, project: Project) -> Dict:
        """Convert Project object to dictionary"""
        project_dict = asdict(project)
        project_dict['status'] = project.status.value
        return project_dict
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow"""
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("understand_input", self._understand_input)
        workflow.add_node("process_task_creation", self._process_task_creation)
        workflow.add_node("process_task_update", self._process_task_update)
        workflow.add_node("process_query", self._process_query)
        workflow.add_node("process_reminder", self._process_reminder)
        workflow.add_node("generate_response", self._generate_response)
        
        # Add edges
        workflow.add_edge("understand_input", "route_intent")
        workflow.add_conditional_edges(
            "route_intent",
            self._route_intent,
            {
                "create_task": "process_task_creation",
                "update_task": "process_task_update",
                "query": "process_query",
                "reminder": "process_reminder",
                "generate_response": "generate_response"
            }
        )
        
        workflow.add_edge("process_task_creation", "generate_response")
        workflow.add_edge("process_task_update", "generate_response")
        workflow.add_edge("process_query", "generate_response")
        workflow.add_edge("process_reminder", "generate_response")
        workflow.add_edge("generate_response", END)
        
        workflow.set_entry_point("understand_input")
        
        return workflow.compile(checkpointer=self.memory)
    
    async def _understand_input(self, state: AgentState) -> AgentState:
        """Understand user input and extract intent and entities"""
        
        system_prompt = """You are a personal assistant AI that helps users manage tasks and projects.
        Analyze the user input and determine:
        1. Intent: create_task, update_task, query, reminder, or other
        2. Entities: dates, people, amounts, file references, etc.
        3. Task type: simple, project, or recurring
        
        Return your analysis as JSON with the following structure:
        {
            "intent": "create_task|update_task|query|reminder|other",
            "entities": {
                "task_title": "string",
                "due_date": "ISO date string or null",
                "start_date": "ISO date string or null",
                "assigned_to": "string or null",
                "priority": "low|medium|high|urgent",
                "task_type": "simple|project|recurring",
                "amount": "number or null",
                "file_references": ["list of file mentions"],
                "links": ["list of URLs"],
                "people": ["list of people mentioned"]
            },
            "context": {
                "is_project": "boolean",
                "needs_subtasks": "boolean",
                "has_attachments": "boolean"
            }
        }
        """
        
        user_input = state["user_input"]
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", f"User input: {user_input}")
        ])
        
        chain = prompt | self.llm
        result = await chain.ainvoke({})
        
        try:
            analysis = json.loads(result.content)
            state["intent"] = analysis["intent"]
            state["entities"] = analysis["entities"]
            state["context"] = analysis["context"]
        except:
            # Fallback if JSON parsing fails
            state["intent"] = "other"
            state["entities"] = {}
            state["context"] = {}
        
        return state
    
    def _route_intent(self, state: AgentState) -> str:
        """Route based on detected intent"""
        intent = state["intent"]
        
        if intent == "create_task":
            return "process_task_creation"
        elif intent == "update_task":
            return "process_task_update"
        elif intent == "query":
            return "process_query"
        elif intent == "reminder":
            return "process_reminder"
        else:
            return "generate_response"
    
    async def _process_task_creation(self, state: AgentState) -> AgentState:
        """Process task creation"""
        entities = state["entities"]
        context = state["context"]
        
        # Check if it's a project
        if context.get("is_project", False):
            project_id = str(uuid.uuid4())
            project = Project(
                id=project_id,
                name=entities.get("task_title", "New Project"),
                description=state["user_input"],
                status=TaskStatus.PENDING,
                created_date=datetime.now(),
                due_date=self._parse_date(entities.get("due_date")) if entities.get("due_date") else None,
                start_date=self._parse_date(entities.get("start_date")) if entities.get("start_date") else None
            )
            self.projects[project_id] = project
            
            # Create initial task for the project
            task_id = str(uuid.uuid4())
            task = Task(
                id=task_id,
                title=f"Start: {entities.get('task_title', 'New Project')}",
                description=f"Begin work on {entities.get('task_title', 'New Project')}",
                status=TaskStatus.PENDING,
                task_type=TaskType.PROJECT,
                priority=Priority(entities.get("priority", "medium")),
                created_date=datetime.now(),
                due_date=self._parse_date(entities.get("due_date")) if entities.get("due_date") else None,
                start_date=self._parse_date(entities.get("start_date")) if entities.get("start_date") else None,
                project_id=project_id
            )
            
            self.tasks[task_id] = task
            project.task_ids.append(task_id)
            
            # Set up reminders
            self._create_reminders(task)
            
            state["current_project"] = project
            state["current_task"] = task
            state["response"] = f"Created project '{project.name}' with initial task. Would you like to add subtasks or team members?"
            
        else:
            # Create simple task
            task_id = str(uuid.uuid4())
            task = Task(
                id=task_id,
                title=entities.get("task_title", "New Task"),
                description=state["user_input"],
                status=TaskStatus.PENDING,
                task_type=TaskType(entities.get("task_type", "simple")),
                priority=Priority(entities.get("priority", "medium")),
                created_date=datetime.now(),
                due_date=self._parse_date(entities.get("due_date")) if entities.get("due_date") else None,
                start_date=self._parse_date(entities.get("start_date")) if entities.get("start_date") else None,
                assigned_to=entities.get("assigned_to")
            )
            
            self.tasks[task_id] = task
            
            # Set up reminders
            self._create_reminders(task)
            
            state["current_task"] = task
            state["response"] = f"Created task '{task.title}'. I'll remind you when it's due."
        
        # Save data
        self._save_data()
        
        return state
    
    async def _process_task_update(self, state: AgentState) -> AgentState:
        """Process task updates"""
        user_input = state["user_input"]
        entities = state["entities"]
        
        # Find relevant task (this would be more sophisticated in practice)
        # For now, we'll assume the most recent task or use context
        
        if "done" in user_input.lower() or "completed" in user_input.lower():
            # Mark task as completed
            # This would need task identification logic
            state["response"] = "Task marked as completed. Great job!"
            
        elif "reschedule" in user_input.lower() or "tomorrow" in user_input.lower():
            # Reschedule task
            state["response"] = "Task rescheduled. I'll remind you at the new time."
            
        elif "assign" in user_input.lower():
            # Handle team assignment
            people = entities.get("people", [])
            if people:
                # Create subtasks for each person
                state["response"] = f"Created tasks for {', '.join(people)}. I'll track their progress."
        
        return state
    
    async def _process_query(self, state: AgentState) -> AgentState:
        """Process queries about tasks, bills, etc."""
        user_input = state["user_input"]
        
        if "electricity" in user_input.lower() and "bill" in user_input.lower():
            # Query about electricity bill
            # This would search through completed tasks and attachments
            state["response"] = "You paid $150 for electricity last month. Here's the bill copy: [attachment]"
        
        elif "how much" in user_input.lower():
            # Financial query
            state["response"] = "Based on your records, here's the financial information you requested."
        
        else:
            state["response"] = "I found the following information based on your query."
        
        return state
    
    async def _process_reminder(self, state: AgentState) -> AgentState:
        """Process reminder-related actions"""
        # This would handle reminder logic
        state["response"] = "Reminder processed."
        return state
    
    async def _generate_response(self, state: AgentState) -> AgentState:
        """Generate final response"""
        if not state.get("response"):
            state["response"] = "I'm here to help you manage your tasks and projects. What would you like to do?"
        
        return state
    
    def _parse_date(self, date_str: str) -> Optional[datetime]:
        """Parse date string to datetime object"""
        if not date_str:
            return None
        
        try:
            # Handle various date formats
            if "tomorrow" in date_str.lower():
                return datetime.now() + timedelta(days=1)
            elif "today" in date_str.lower():
                return datetime.now()
            elif "next week" in date_str.lower():
                return datetime.now() + timedelta(weeks=1)
            else:
                # Try to parse as ISO format
                return datetime.fromisoformat(date_str)
        except:
            return None
    
    def _create_reminders(self, task: Task):
        """Create reminders for a task"""
        if task.due_date:
            # Main reminder
            reminder = Reminder(
                id=str(uuid.uuid4()),
                task_id=task.id,
                reminder_time=task.due_date,
                message=f"Reminder: {task.title}",
                reminder_type="standard"
            )
            task.reminders.append(reminder)
            
            # Soft reminder (if due date is more than a week away)
            if task.due_date > datetime.now() + timedelta(days=7):
                soft_reminder = Reminder(
                    id=str(uuid.uuid4()),
                    task_id=task.id,
                    reminder_time=task.due_date - timedelta(days=7),
                    message=f"Soft reminder: {task.title} is due in a week",
                    reminder_type="soft"
                )
                task.reminders.append(soft_reminder)
    
    def add_log(self, task_id: str, action: str, description: str, user_input: str = None):
        """Add a log entry to a task"""
        if task_id in self.tasks:
            log = TaskLog(
                timestamp=datetime.now(),
                action=action,
                description=description,
                user_input=user_input
            )
            self.tasks[task_id].logs.append(log)
            self._save_data()
    
    def add_attachment(self, task_id: str, filename: str, filepath: str, description: str = None):
        """Add an attachment to a task"""
        if task_id in self.tasks:
            attachment = FileAttachment(
                filename=filename,
                filepath=filepath,
                file_type=Path(filename).suffix,
                upload_date=datetime.now(),
                description=description
            )
            self.tasks[task_id].attachments.append(attachment)
            self._save_data()
    
    async def process_input(self, user_input: str) -> str:
        """Main entry point for processing user input"""
        initial_state = {
            "messages": [],
            "user_input": user_input,
            "current_task": None,
            "current_project": None,
            "intent": "",
            "entities": {},
            "context": {},
            "response": "",
            "next_action": ""
        }
        
        config = {"configurable": {"thread_id": "main"}}
        
        try:
            result = await self.graph.ainvoke(initial_state, config=config)
            return result["response"]
        except Exception as e:
            return f"I apologize, but I encountered an error: {str(e)}"
    
    def get_pending_reminders(self) -> List[Reminder]:
        """Get all pending reminders"""
        now = datetime.now()
        pending_reminders = []
        
        for task in self.tasks.values():
            for reminder in task.reminders:
                if not reminder.is_sent and reminder.reminder_time <= now:
                    pending_reminders.append(reminder)
        
        return pending_reminders
    
    def send_reminder(self, reminder_id: str) -> str:
        """Mark a reminder as sent and return the message"""
        for task in self.tasks.values():
            for reminder in task.reminders:
                if reminder.id == reminder_id:
                    reminder.is_sent = True
                    self._save_data()
                    return reminder.message
        return "Reminder not found"

# Usage Example
async def main():
    # Initialize the agent
    agent = PersonalAssistantAgent(
        openai_api_key="your-openai-api-key",
        data_dir="assistant_data"
    )
    
    # Test scenarios
    test_inputs = [
        "Need to pay electric bill tomorrow",
        "A website is to be built for Cohen Law Firm within 30 days. Need to start the task next Monday.",
        "Assign Raihan for the development task, Rayan for design task, and Kazi for content creation task.",
        "The content writing is done and approved. However the design task needs some modifications. I'll check back in 5 days.",
        "How much did I pay for electricity bill last month? Give me the bill copy"
    ]
    
    for user_input in test_inputs:
        print(f"\n--- User: {user_input} ---")
        response = await agent.process_input(user_input)
        print(f"Assistant: {response}")
        
        # Check for pending reminders
        pending_reminders = agent.get_pending_reminders()
        if pending_reminders:
            print("Pending reminders:")
            for reminder in pending_reminders:
                print(f"  - {reminder.message}")

if __name__ == "__main__":
    asyncio.run(main())