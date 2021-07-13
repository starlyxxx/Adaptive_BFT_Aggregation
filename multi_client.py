from fl_client import FederatedClient
import ea_datasource
import datasource
import multiprocessing
import threading
from src.parsingconfig import readconfig
import load_car10

def start_client(cid,ip,port,thread):
    print("start client")
    c = FederatedClient(ip, int(port), thread, cid) 

if __name__ == '__main__':
    nodes, servers, clients, baseport, LOCAL, ip_address = readconfig(0)
    jobs = []
    # cid for clientID, start from 0
    for i in range(int(nodes)//int(clients)):
        # threading.Thread(target=start_client).start()
        
        #thread = load_car10.Car10()
        thread = datasource.Mnist()
        p = multiprocessing.Process(target=start_client,args=(str(i),ip_address,baseport,thread))
        jobs.append(p)
        p.start()
    # TODO: randomly kill