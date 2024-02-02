import streamlit as st

def main():
    st.title("Birth Information Form")

    # Create form components
    name = st.text_input("Name", "")
    date_of_birth = st.date_input("Date of Birth", format="YYYY/MM/DD")
    place_of_birth = st.text_input("Place of Birth", "")
    time_of_birth = st.time_input("Time of Birth")
    query = st.text_input("ask me", " aspicious time")

    # Add validation for mandatory fields
    if st.button("Submit"):
        if not name or not date_of_birth or not place_of_birth or not time_of_birth or not query:
            st.error("All fields are mandatory. Please fill in all the information.")
        else:
            op = f""" I am {name}, date of birth {date_of_birth}, place of birth {place_of_birth}, time of birth {time_of_birth}. {query}
             """
            st.success(op)
            # Process the form data or perform further actions here

if __name__ == "__main__":
    main()
