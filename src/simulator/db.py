
from sqlalchemy import create_engine
from sqlalchemy.orm import scoped_session, sessionmaker

engine = create_engine('sqlite:///:memory:', echo=True)
Session = scoped_session(sessionmaker(bind=engine))