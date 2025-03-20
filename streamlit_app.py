import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from langchain_core.messages import HumanMessage, AIMessage
import re

# Set page configuration
st.set_page_config(
    page_title="Data Chat Assistant",
    layout="wide"
)

# Sample data (in a real app, this would connect to your data lake)
@st.cache_data
def load_sample_data():
    # Sales data
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
    sales = np.random.normal(1000, 200, len(dates))
    sales = np.cumsum(np.random.normal(0, 100, len(dates))) + sales  # Add trend
    
    # Add seasonality
    for i in range(len(dates)):
        # Weekly pattern
        sales[i] += 200 * np.sin(i * 2 * np.pi / 7)
        # Monthly pattern
        sales[i] += 400 * np.sin(i * 2 * np.pi / 30)
    
    sales_df = pd.DataFrame({
        'date': dates,
        'sales': sales,
        'category': np.random.choice(['Electronics', 'Clothing', 'Food', 'Home Goods'], len(dates)),
        'region': np.random.choice(['North', 'South', 'East', 'West'], len(dates)),
    })
    
    # Customer data
    customer_df = pd.DataFrame({
        'customer_id': range(1000),
        'age': np.random.normal(35, 10, 1000).astype(int),
        'gender': np.random.choice(['M', 'F', 'Other'], 1000),
        'total_spent': np.random.exponential(1000, 1000),
        'loyalty_years': np.random.exponential(3, 1000),
        'preferred_category': np.random.choice(['Electronics', 'Clothing', 'Food', 'Home Goods'], 1000),
    })
    
    return sales_df, customer_df

sales_df, customer_df = load_sample_data()

# Simulated NLP function to interpret user queries
def process_query(query, chat_history):
    """
    Process natural language query to determine what visualization to show.
    In a production system, this would use a real NLP model.
    """
    query = query.lower()
    
    # Look for time-related patterns
    if any(term in query for term in ['daily', 'day by day', 'over time', 'trend']):
        return {
            'type': 'time_series',
            'title': 'Sales Over Time',
            'explanation': "Here's the daily sales trend. I've added a 7-day moving average to help identify patterns."
        }
    
    # Look for category comparison patterns
    if any(term in query for term in ['category', 'categories', 'compare', 'breakdown']):
        return {
            'type': 'category_breakdown',
            'title': 'Sales by Category',
            'explanation': "Here's the breakdown of sales by product category. Electronics leads with the highest revenue."
        }
    
    # Look for regional analysis
    if any(term in query for term in ['region', 'regional', 'location', 'where']):
        return {
            'type': 'regional',
            'title': 'Regional Sales Performance',
            'explanation': "I've analyzed the sales by region. The West region shows the highest performance."
        }
    
    # Look for customer analysis
    if any(term in query for term in ['customer', 'customers', 'buyers', 'demographic']):
        return {
            'type': 'customer',
            'title': 'Customer Demographics',
            'explanation': "Here's an analysis of customer demographics including age distribution and spending patterns."
        }
    
    # Default response
    return {
        'type': 'default',
        'title': 'Sales Summary',
        'explanation': "I'm showing a general overview of sales data. You can ask for specific analyses like trends over time, category breakdowns, or regional performance."
    }

# Create visualizations based on the query type
def create_visualization(query_result):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if query_result['type'] == 'time_series':
        # Time series plot
        df = sales_df.copy()
        df['7day_avg'] = df['sales'].rolling(window=7).mean()
        
        ax.plot(df['date'], df['sales'], alpha=0.5, label='Daily Sales')
        ax.plot(df['date'], df['7day_avg'], linewidth=2, label='7-day Moving Average')
        ax.set_xlabel('Date')
        ax.set_ylabel('Sales Amount')
        ax.set_title(query_result['title'])
        ax.legend()
        
    elif query_result['type'] == 'category_breakdown':
        # Category breakdown
        category_sales = sales_df.groupby('category')['sales'].sum().sort_values(ascending=False)
        category_sales.plot(kind='bar', ax=ax)
        ax.set_xlabel('Product Category')
        ax.set_ylabel('Total Sales')
        ax.set_title(query_result['title'])
        
    elif query_result['type'] == 'regional':
        # Regional analysis
        region_sales = sales_df.groupby('region')['sales'].sum().sort_values(ascending=False)
        region_sales.plot(kind='bar', ax=ax, color=sns.color_palette("viridis", 4))
        ax.set_xlabel('Region')
        ax.set_ylabel('Total Sales')
        ax.set_title(query_result['title'])
        
    elif query_result['type'] == 'customer':
        # Customer analysis - create a subplot with 2 charts
        plt.close(fig)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Age distribution
        sns.histplot(customer_df['age'], kde=True, ax=ax1)
        ax1.set_title('Customer Age Distribution')
        ax1.set_xlabel('Age')
        
        # Spending by category
        cat_spending = customer_df.groupby('preferred_category')['total_spent'].sum().sort_values(ascending=False)
        cat_spending.plot(kind='pie', ax=ax2, autopct='%1.1f%%')
        ax2.set_title('Spending by Preferred Category')
        ax2.set_ylabel('')
        
        plt.tight_layout()
        
    else:
        # Default visualization - general summary
        sales_by_region_cat = sales_df.pivot_table(
            values='sales', 
            index='region', 
            columns='category', 
            aggfunc='sum'
        )
        sales_by_region_cat.plot(kind='bar', stacked=True, ax=ax)
        ax.set_xlabel('Region')
        ax.set_ylabel('Total Sales')
        ax.set_title('Sales by Region and Category')
        ax.legend(title='Category')
    
    plt.tight_layout()
    return fig

# Initialize session state for chat history if it doesn't exist
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="Hello! I'm your data assistant. Ask me to create dashboards or reports from your data.")
    ]

# Display header
st.title("💬 Data Chat Assistant")
st.caption("Ask questions about your data to generate dashboards")

# Display chat messages
for message in st.session_state.chat_history:
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.write(message.content)
    else:
        with st.chat_message("assistant"):
            st.write(message.content)
            
            # Display visualizations for AI messages that have them
            if hasattr(message, 'visualization') and message.visualization:
                st.pyplot(message.visualization)

# Chat input
user_query = st.chat_input("Ask about your data...")
if user_query:
    # Add user message to chat
    st.session_state.chat_history.append(HumanMessage(content=user_query))
    
    # Display user message
    with st.chat_message("user"):
        st.write(user_query)
    
    # Process the query
    with st.spinner("Generating insights..."):
        # Get query result
        query_result = process_query(user_query, st.session_state.chat_history)
        
        # Create visualization
        fig = create_visualization(query_result)
        
        # Create AI response
        ai_message = AIMessage(content=query_result['explanation'])
        ai_message.visualization = fig  # Attach visualization to message
        
        # Add AI message to chat history
        st.session_state.chat_history.append(ai_message)
        
        # Display AI message with visualization
        with st.chat_message("assistant"):
            st.write(query_result['explanation'])
            st.pyplot(fig)

# Sidebar with data preview
with st.sidebar:
    st.header("Data Preview")
    
    st.subheader("Sales Data")
    st.dataframe(sales_df.head(), use_container_width=True)
    
    st.subheader("Customer Data")
    st.dataframe(customer_df.head(), use_container_width=True)
    
    st.caption("Note: This is sample data. In a production system, this would connect to your data lake.")
    
    # Example queries
    st.subheader("Example Queries")
    example_queries = [
        "Show me the sales trend over time",
        "Break down sales by category",
        "How are sales distributed by region?",
        "Analyze customer demographics"
    ]
    
    for query in example_queries:
        if st.button(query):
            # This will be processed when the page refreshes
            st.session_state.example_query = query

# Handle example query buttons
if 'example_query' in st.session_state:
    query = st.session_state.example_query
    del st.session_state.example_query  # Clear after use
    
    # Add user message to chat
    st.session_state.chat_history.append(HumanMessage(content=query))
    
    # Process the query
    query_result = process_query(query, st.session_state.chat_history)
    fig = create_visualization(query_result)
    
    # Create AI response
    ai_message = AIMessage(content=query_result['explanation'])
    ai_message.visualization = fig
    
    # Add AI message to chat history
    st.session_state.chat_history.append(ai_message)
    
    # Force a rerun to display the new messages
    st.rerun()
