import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="Prompt Config ",
    page_icon="",
    layout="wide",
)

st.markdown("""
<style>
.stTextArea>div>div>textarea {
    font-size: 18px; 
    height: 100px; 
}
</style>
""", unsafe_allow_html=True)
st.header("Please configure youre prompt expert!")

def create_table():
    """
    Creates a table with add and remove row functionality.
    """

    data = []
    if "data" not in st.session_state:
        st.session_state.data = data

    
    # First row - Text input for "Prompt Name"
    pn=st.text_input("Prompt Name", key="col1")

    # Second row - Text input for "Description"
    des=st.text_input("Description", key="col2")

    # Third row - Text area for "Enter your complete text box here"
    prompt=st.text_area("Enter your complete prompt here:", height=150,key="col3")

    
    if st.button("Add Row"):
        new_row = [pn, des, prompt]
        st.session_state.data.append(new_row)
    table_df=pd.DataFrame(st.session_state.data,columns= ['Prompt Name', 'Description',"complete prompt"])
    st.table(table_df)
     
    for i, row in enumerate(st.session_state.data):
        cols = st.columns([1, 1])
        with cols[1]:
            if st.button(f"Remove Row {i+1}"):
                st.session_state.data.pop(i)
                break  

create_table()

def convert_to_dataframe():
  """Converts data from st.session_state.data to a pandas DataFrame."""
  if 'data' in st.session_state:
    df = pd.DataFrame(st.session_state["data"])
    st.write("Dataframe created successfully!")
    st.dataframe(df) 
    df.columns = ["name", "description", "prompt_template"]  
    # Save the dataframe to a JSON file
    df.to_json("saved_data.json", orient="records") 
    st.success("Data saved to saved_data.json")
  else:
    st.write("Data does not exist in st.session_state")

# Create a button to trigger the conversion and saving
if st.button("save prompt template to json"):
  convert_to_dataframe()