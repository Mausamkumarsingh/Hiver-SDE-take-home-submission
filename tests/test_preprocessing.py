import pytest
from src.data.preprocessor import clean_text, is_predominantly_english, extract_resolution_links

def test_clean_text_removes_handles_and_html():
    raw = "@AmazonHelp &amp; @115820 Where is my package??? &gt;&lt;"
    cleaned = clean_text(raw, is_brand_reply=False)
    assert "@AmazonHelp" not in cleaned
    assert "@115820" not in cleaned
    assert "&" in cleaned
    assert ">" in cleaned

def test_clean_text_strips_agent_signatures():
    raw_brand = "@115820 Please visit https://t.co/xyz for tracking info. ^TN"
    cleaned = clean_text(raw_brand, is_brand_reply=True)
    assert "^TN" not in cleaned
    assert "https://t.co/xyz" in cleaned

def test_is_predominantly_english():
    assert is_predominantly_english("Where is my order? It was supposed to arrive today.") is True
    assert is_predominantly_english("こんにちは、アマゾン公式です。") is False
    assert is_predominantly_english("Guten Tag, wie kann ich Ihnen helfen?") is False

def test_extract_resolution_links():
    text = "Please reach out to us here: https://amazon.com/help or https://t.co/test"
    links = extract_resolution_links(text)
    assert len(links) == 2
    assert "https://amazon.com/help" in links
