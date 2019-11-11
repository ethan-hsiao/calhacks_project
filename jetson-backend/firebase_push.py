import firebase_admin
from firebase_admin import credentials
from firebase_admin import db

class DBPusher():
    def __init__(self):
        self.cred = credentials.Certificate('jetson-firebase-key.json')
        self.default_app = firebase_admin.initialize_app(self.cred, {
        'databaseURL': 'https://calhacks-cv.firebaseio.com/'
        })
    
    def push(self, num_ppl, lines, clusters):

        clust_str = ""
        for cluster in clusters:
            clust_str += str(cluster[0]) + " " + str(cluster[1]) + " " + str(cluster[2]) + " " + str(cluster[3]) + ','
        
        line_str = ""
        for line in lines:
            points_str = ''
            for p in line:
                points_str += str(p[0]) + '-' + str(p[1]) + " "
            line_str += points_str[:-1]+","
        db.reference('/clusters').push(clust_str[:-1])
        db.reference('/num_ppl').push(str(num_ppl))
        db.reference('/lines').push(line_str[:-1])
