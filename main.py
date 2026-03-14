import ollama
import streamlit
import streamlit as st
print(f"ollama当前可用的模型有：f{ollama.list()}")
print(f"streamlit库的版本是：{streamlit.__version__}")