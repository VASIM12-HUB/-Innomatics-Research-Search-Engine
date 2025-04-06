from chromadb import PersistentClient

client = PersistentClient(path="db")
collection = client.get_or_create_collection(name="subtitles")