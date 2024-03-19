# user_manager.py
from sqlalchemy import create_engine, Column, Integer, String, Boolean, Sequence
from sqlalchemy.orm import declarative_base,sessionmaker, Session

DATABASE_URL = "sqlite:///./base_utilisateur.db" #  la base de données SQLite sera stockée dans le fichier base_utilisateur.db dans le répertoire actuel (./).

engine = create_engine(DATABASE_URL) # Le moteur est responsable de la communication avec la base de données. 

Base = declarative_base() #La classe de base déclarative fournit des fonctionnalités de base pour déclarer des modèles de manière élégante.
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
#Définier la sessions de base de données. 
db = SessionLocal()
def get_db():
    db = SessionLocal()
    try:
        print("Session created successfully.")
        #yield db
        return db
        #print("Session created successfully.") 
    finally:
        db.close()

class User(Base):
    __tablename__ = 'users'

    id = Column(Integer, Sequence('user_id_seq'), primary_key=True)
    username = Column(String(50), unique=True, index=True)
    password = Column(String(50))
    admin = Column(Boolean, default=False)


Base.metadata.create_all(bind=engine)

#SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
# Creating initial users

def create_initial_users():
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    db = SessionLocal()

    users = [
        User(username='admin1', password='adminpass1', admin=True),
        User(username='Kani', password='Kanipass1', admin=False),
        User(username='Hassan', password='Hassanpass1', admin=False),
    ]

    db.add_all(users)
    db.commit()
    db.close()

def create_user(db, username, password):
    user = User(username=username, password=password,admin=False)
    db.add(user)
    db.commit()
    db.refresh(user)
    return user

def get_user_by_username(db, username):
    return db.query(User).filter(User.username == username).first()

def delete_user(db, username):
    user = db.query(User).filter(User.username == username).first()
    if user:
        db.delete(user)
        db.commit()
        return True
    return False
def get_all_users(db: Session):
    print("Executing get_all_users")
    users = db.query(User).all()
    print("Users from the database:", users)
    return users
#if __name__ == "__main__":
 #   create_initial_users()
get_all_users(db)
# Ajoutez d'autres fonctions pour la gestion des utilisateurs au besoin
