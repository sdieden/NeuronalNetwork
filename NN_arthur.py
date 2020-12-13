import numpy as np
import math
import matplotlib.pyplot as plt

def read_csv(fichier) :

  """"
  retourne une matrice de floats du fichier csv sans la ligne de catégorie
  """

  with open(fichier) as fd :
    data = []
    nombre_de_lignes = 0 #pour connaitre la dim de la matrice
    for line in fd :
      ligne = []
      nombre_de_criteres = 0 #meme chose ici
      for number in line.split(sep=";"): 
        ligne.append(number)
        nombre_de_criteres +=1
      data.append(ligne)
      nombre_de_lignes += 1
    
    data = np.array(data).reshape(nombre_de_lignes,nombre_de_criteres)
    data = data[1::,:] #retire la 1ere ligne de strings
    data = np.array(data, dtype = float)
    return data

def boolien(v): 
  """ 
  prend une valeur et retourne sa valeur boolien, si c'est pas de type boolien envoie une erreur
  """
  if isinstance(v, bool):
    return v
  elif v.lower() in ('yes', 'true', 't', 'y', '1','oui','o','vrai','v'):
    return True
  elif v.lower() in ('no', 'false', 'f', 'n', '0','non','n','faux','f'):
    return False
  else:
    raise ValueError ("-----------------ERROR--------------- \n!!! type d'activation invalide, valeur booléen attendue !!!\n-----------------ERROR---------------")

def square (x) :
  """
  met tout les elements d'une liste au carré
  """
  if isinstance(x,list) :
    return np.array([i ** 2 for i in x])
  else :
    raise ValueError("-----------------ERROR--------------- \n!!! type d'activation invalide, une valeur de type 'list' est attendue !!!\n-----------------ERROR---------------")


class NetworkNeurons :
  """
  cette classe permet d'utiliser le procedé du machine learning pour prédire des données
  """
  A = None
  B = None
  C = None

  def __init__ (self, matrix_data, neurons_per_line, nombres_de_lignes_de_neurones = 1, 
          learning_rate = 0.01, activation_interne_de_type_ReLu = False, 
          pourcentage_de_data_pour_le_training = 0.8,random_sous_forme_de_gaussienne = False, epoque = 0) :
    """
    Initalise la classe avec plusieurs arguments :

      matrix_data : est la matrice numpy des données utiliser pour configurer le programme avec en derniere colonne les outputs recherchées
      neurons_per_line : est le nombre de neurones désirer par ligne
      nombres_de_lignes_de_neurones : est le nombre de ligne souhaité (defaut 1)
      learning_rate : est une valeur qui défini la vitesse d'apprentissage (defaut 0.01)
      activation_interne_de_type_ReLu : est un parametre qui défini si entre les couches de neurones (sauf pour la derniere) on désire utiliser la fonction ReLU plustot que la sigmoid (defaut = False)
      pourcentage_de_data_pour_le_training : Défini le pourcentage de data utilisée pour l'entrainement (defaut = 0.8)
      epoque = défini combien de fois on a fait tourner avec le meme set de data

    les variables créer dans __init__ on pour objectif :

      permetre au autres fonctions de la classe d'utiliser les arguments de __init__
      et ou 
        self.args donne le nombre de variables initiales
        self.liste_erreurs_... creer la liste de differance entre la vrai note du vin et notre note calculée
        self...._n_data donne exactement le nombre de echantillions (ici du vin) alloué au training ou a la vérification

    Lors de la 1ere epoque on va creer 

    """

    self.data = matrix_data
    self.n_neurons = int(neurons_per_line)
    self.n_lignes = int(nombres_de_lignes_de_neurones) 
    self.lr = float(learning_rate)
    self.bool = boolien(activation_interne_de_type_ReLu)
    self.args = (len(self.data[0])-1)#on ne prend pas l'agument note
    self.liste_erreurs_train = []
    self.liste_erreurs_verif = []
    self.gaus = boolien(random_sous_forme_de_gaussienne)
    self.epoque = epoque


    if pourcentage_de_data_pour_le_training < 1 : #au cas ou on met comme argument du pourcentage un nombre qui est entre 0 et 100 au lieu d'etre entre 0 et 1
      self.train_n_data = int(round(pourcentage_de_data_pour_le_training * len(self.data) , 0))
    else :
      self.train_n_data = int(round((pourcentage_de_data_pour_le_training/100) * len(self.data), 0))

    self.verif_n_data = int(len(self.data) - self.train_n_data)

      #_________CREATIONS INITIALE DES MATRICES DE POIDS___________#


    if epoque == 0 : #pour pas recréer des martices random a chaque génération

      if self.gaus :

        self.__class__.A = (
          np.random.normal(size= self.n_neurons * self.args)
            .reshape(1,self.n_neurons,self.args) #le 1re argument dans .reshape et 0 pour que la matrice soit 3D
        )

        if self.n_lignes > 1 : #afin d'éviter d'avoir une erreur de type np.reshape(0,X,Y) est impossible
          
          self.__class__.B = (
            np.random.normal(size=self.n_neurons*self.n_neurons*(self.n_lignes - 1))
            .reshape(self.n_lignes -1,self.n_neurons,self.n_neurons)
          )
        #matrice carré 3D des poids internes et il y a n**2 * (l-1) elements ou n et l sont le nombre de neurones et l le nombre de lignes
        
        self.__class__.C = (
          np.random.normal(size = self.n_neurons)
          .reshape(1,1,self.n_neurons)
        )

      else :
        self.__class__.A = (
          np.random.uniform(-1,1,size= self.n_neurons * self.args)
            .reshape(1,self.n_neurons,self.args) #le 1re argument dans .reshape et 0 pour que la matrice soit 3D
        )

        if self.n_lignes > 1 : #afin d'éviter d'avoir une erreur de type np.reshape(0,X,Y) est impossible
          
          self.__class__.B = (
            np.random.uniform(-1,1,size=self.n_neurons*self.n_neurons*(self.n_lignes - 1))
            .reshape(self.n_lignes -1,self.n_neurons,self.n_neurons)
          )
        #matrice carré 3D des poids internes et il y a n**2 * (l-1) elements ou n et l sont le nombre de neurones et l le nombre de lignes
        
        self.__class__.C = (
          np.random.uniform(-1,1,size = self.n_neurons)
          .reshape(1,1,self.n_neurons)
        )
     



  def linear (self,matrice_des_poids,neurone = 0, ligne = 0) :
    """
    fait le produit scalire (ou combinaison linéaire) 
    du vecteur input (soit les inputs initiaux soit les outputs de la rangé pécdente de neurones)
    avec la matrice 3D de poids qui est soit A soit B soit C

      A = fausse matrice 3D car elle est 2D mais de longeur 0 dans la 3e dim et elle represente les poids de la 1er ligne de neurones
      B = matrice 3D des poids de tout les autres neurones
      C = vecteur des poids de l'output

    NB. !!! toutes les matrices doivent etre 3D pour avoir une bonne indexation !!!
    """
    prod_scal = 0
    for i in range(len(self.input)) :
      prod_scal += self.input[i] * matrice_des_poids[ligne,neurone,i] 
        #ou matrice_des_poids est la matrice de 3D (nombres de lignes, nombres neurones par lignes, nombres de criters) de poids (weight)
        # avec l la ligne sur la quelle on travaille, n le neurone sur lequel on travaille et i l'input en question
      return prod_scal
    

  def activation_sigmoid(self,x):
      '''
      passe le produit scalaire par la fonction d'activation sigmoid où
      S(x) = 1/(1+e^(-x))
      '''
      try :
        output = 1/(1+math.exp(-x))
        return output
      except OverflowError :

        if x>0 :
          return 1
        elif x< 0 :
          return 0

      

  def activation_ReLU (self,x) :
    """
    passe le produit scalaire par la fonction d'activation ReLU où

    ReLU(x) = { x si x >= 0
              { 0 si x < 0
    """
    if x < 0 :
      return 0
    else :
      return x

  def derivate_activation_sigmoid(self,x):
    """
    donne la valeur de x par la deivé de la fontion d'activation sigmoid (S(x)) où 
      
      d/dx(S(x)) = S(x)*(1-S(x)) 

    """
    sortie = self.activation_sigmoid(x) * (1 - self.activation_sigmoid(x))
    return sortie

  def derivate_activation_ReLU (self,x) :
    """
    donne la valeur de x par la deivé de la fontion d'activation ReLU 
    avec la particularité qui est que arbitrairement on a choisi 0.5 comme valeur de la dérivé en 0
    """
    if x < 0 :
      return 0
    elif x > 0 :
      return 1
    else :
      return 0.5 
      #puisque la dérivé est differante si on prend a droite ou a gauche on vas dire qu'en 0 (ce qui statistiquement n'arrivera probablement jamais) la dérivé vaut 0.5

  def calculations (self,vin) : 
    """
    retourne une matrice des outputs des neurones de toutes les lignes pour un vin et la note calculée du vin.
    Rajoute aussi la differance entre la vrai note et notre note dans la liste_erreurs correspondante (training ou verification)
    """
    self.input = self.data[vin,:self.args] #prends tout les inputs initaux 
    output_int = [] #défini une liste d'outputs interne
    
    for y in range(self.n_neurons): #ligne 1
      if self.bool :
        output_int.append(
          self.activation_ReLU(self.linear(self.__class__.A,y))
        )
      else :
        output_int.append(
          self.activation_sigmoid(self.linear(self.__class__.A,y))
        )
    self.input = output_int

    
    for x in range(self.n_lignes-1) : #toutes les autres lignes
      
      output = []
      for y in range(self.n_neurons):

        if self.bool :
          output.append(self.activation_ReLU(self.linear(self.__class__.B,y,x)))
          output_int.append(self.activation_ReLU(self.linear(self.__class__.B,y,x)))
            

        else :
          output.append(self.activation_sigmoid(self.linear(self.__class__.B,y,x)))
          output_int.append((self.activation_sigmoid(self.linear(self.__class__.B,y,x))))
        
      self.input = output

            
    output_int = np.array(output_int).reshape(self.n_lignes,self.n_neurons) #les neurones sont a l'horizontal dans la matrice !

    output_end = 10 * self.activation_sigmoid(self.linear(self.__class__.C))

    if vin in range(self.train_n_data) : #permet de remplir soit la liste d'erreur verif soit celle du training
      self.liste_erreurs_train.append(self.data[vin,-1] - output_end) #créer la liste de differances entre la vrai note et la note calculée

    else :
      self.liste_erreurs_verif.append(self.data[vin,-1] - output_end)

    return output_int , output_end #retourne un tuple avec en 0 la matrice et en 1 la note du vin

    
  def errors (self,vin) :
    """
    Trouve les ereurs sur tout les poids par rapport a l'erreur sur la note du vin et 
    ajoute à la liste self.list_erreurs la differance entre la vrai note et la note calculée
    """
    output_int , output_end = self.calculations(vin) 
    error = np.zeros(self.n_lignes*self.n_neurons).reshape(self.n_lignes,self.n_neurons)
    y_true = self.data[vin,-1]
    
    final_error = (y_true - output_end) * self.derivate_activation_sigmoid(output_end)
    
    if self.bool :
      for i in range(self.n_neurons): #derniere ligne de neurones donc on utilise final_error
        error[self.n_lignes - 1 , i] += self.__class__.C[0,0,(self.n_neurons-1)-i] * final_error * self.derivate_activation_ReLU(final_error)
        #ici faire self.n_neurone -1 permet de s'occuper du probleme que l'indexation de matrice commence à O 

      if self.n_lignes > 1 :

        for line in range(self.n_lignes) :

          for neuron in range (self.n_neurons) :
            somme = 0
            for poids in range(self.n_neurons) :
              somme += self.__class__.B[(self.n_lignes-2) - line , (self.n_neurons-1)-neuron,poids]* error[(self.n_lignes-1) - line ,poids]
            error[(self.n_lignes-1) - line , neuron] += somme * self.derivate_activation_ReLU(output_int[(self.n_lignes-1) - line , neuron])
      
    else :
      for i in range(self.n_neurons):
        error[self.n_lignes - 1 , i] += (self.__class__.C[0,0,(self.n_neurons-1)-i] * final_error * self.derivate_activation_sigmoid(final_error))

      if self.n_lignes > 1 :

        for line in range(self.n_lignes) :

          for neuron in range (self.n_neurons) :
            somme = 0
            for poids in range(self.n_neurons) :
              somme += self.__class__.B[(self.n_lignes-2) - line , (self.n_neurons-1)-neuron,poids]* error[(self.n_lignes-1) - line ,poids]
            error[(self.n_lignes-1) - line , neuron] += somme * self.derivate_activation_sigmoid (output_int[(self.n_lignes-1) - line , neuron])

    return error , final_error

  
  def weight_update (self,vin) :
    """
    redéfini tout les poids des 3 matrices pour que nos notes calculées soit plus proche de la valeur recherché 
    """
    output_int = self.calculations(vin)[0] 
    error , final_error = self.errors(vin)

    for neurone in range(self.n_neurons) :

      self.__class__.C[0,0,neurone] = self.__class__.C[0,0,neurone] + self.lr * final_error * output_int[self.n_lignes -1,neurone]

      for poids in range(self.args) :

        self.__class__.A[0,neurone ,poids] = self.__class__.A[0,neurone ,poids] + self.lr * error[0,neurone] * self.data[vin,poids]

        for line in range (self.n_lignes-1) :

          for x in range(self.n_neurons) :

            self.__class__.B[line,neurone,x] = self.__class__.B[line,neurone,x] + self.lr * error[line +1 ,neurone] * output_int[line,neurone]

  def make_it_happen (self) :
    """
    calcule tout le fichier csv (training + verification)
    et renvoie les données d'erreur de l'époque
    """

    for vin in range(len(self.data)) :
      
      if vin in range(self.train_n_data) :
        self.weight_update(vin)
      else :
        self.calculations(vin) #pas besoin de faire toute la backpropagation pour la verification
    
    return self.liste_erreurs_train, self.liste_erreurs_verif

   

if __name__ == "__main__":

  def MSE (nombre_d_epoques) : 

    """
    Calcule le MSE de la Classe sur un nombre de génération choisi 
    Et dessine les 3 graphiques
    """

    MSE_train = np.zeros(nombre_d_epoques)
    MSE_verif = np.zeros(nombre_d_epoques)
    for i in range(nombre_d_epoques) :
      nn = NetworkNeurons(read_csv("projet/winequality-red.csv"),20,1,0.07,epoque= i)
      erreurs_train , erreurs_verif = nn.make_it_happen()
      print('training on generation {} currently working ...{}'.format(i+1,"."*(i%2)+" ",),end = '\r') 
      MSE_train[i] += sum(square(erreurs_train))/(1+i)
      MSE_verif[i] += sum(square(erreurs_verif))/(1+i)

    plt.plot(MSE_train,label = "Training")
    plt.plot(MSE_verif,label = "Verification",color = "red")
    plt.xlabel("epoques")
    plt.ylabel("MSE")
    plt.title("MSE par époque")
    plt.show()
    plt.savefig("MSE")

  MSE(50)

#test

    
    
    

  






    



        
      


  