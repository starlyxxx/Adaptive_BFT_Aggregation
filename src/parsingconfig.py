import configparser
configFile = "./config.ini" 

###########
# general use
############
def readconfig(replicaID):  
    config = configparser.ConfigParser() #create a config parser 
    config.read(configFile)
    return (int(config['summary']['nodes']), int(config['summary']['servers']), int(config['summary']['clients']), int(config['summary']['baseport']), int(config['summary']['LOCAL']), config[str(replicaID)]['ip_address'])

def readclientconfig(replicaID):  
    config = configparser.ConfigParser() #create a config parser 
    config.read(configFile)
    return (int(config['summary']['nodes']), int(config['summary']['servers']), int(config['summary']['clients']), int(config['summary']['baseport']), int(config['summary']['LOCAL']), config[replicaID]['ip_address'])

def readsocketinfo(replicaID):
    config = configparser.ConfigParser() #create a config parser 
    config.read(configFile)
    return (config[str(replicaID)]['ip_address'],int(config[str(replicaID)]['port_number']))

#get the maximum number of client ID
def client_config_default():
    config = configparser.ConfigParser() #create a config parser 
    config.read(configFile)
    curMax = 999 ##By default, client ID can only be greater than 999
    for section in config.sections():
        if not section.isdigit() or int(section) < 999:
            continue
        if int(section) > curMax:
            curMax = int(section)
    return curMax

#check whether a section exists in config.ini
def check_section(clientID):
    config = configparser.ConfigParser() #create a config parser 
    config.read(configFile)
    if config.has_section(str(clientID)):
        return True
    return False

#add in database
def add_config_section(clientID, clientIP, clientPort):
    config = configparser.ConfigParser() #create a config parser 
    if check_section(clientID):
        return
    f = open(configFile, 'a') 
    config.add_section(str(clientID))
    config.set(str(clientID), 'client_id', str(clientPort))
    config.set(str(clientID), 'ip_address', str(clientIP))
    config.write(f)
    f.close()

def getIPFromConfig(cid):
    config = configparser.ConfigParser() #create a config parser 
    config.read(configFile)
    return config[str(cid)]['ip_address']

def readconfig_clientAddr(): ##By default, client ID can only be greater than 999
    clientAddrTable = {}
    config = configparser.ConfigParser() #create a config parser 
    config.read(configFile)
    for section in config.sections():
        if not section.isdigit() or int(section) < 999:
            continue
        clientAddrTable[section] = config[section]['ip_address']
    return clientAddrTable

if __name__ == '__main__':
    print("")
