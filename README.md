# School_of_AI-Assignment_13


# Part 1: Unet 

First part of our assignment is to train our own UNet from scratch,

case 1: MP+Tr+BCE

<img width="513" alt="image" src="https://user-images.githubusercontent.com/63030539/235317043-1debb321-091b-4be4-ac9d-4864cd61012c.png">


case 2: MP+Tr+Dice Loss

<img width="509" alt="image" src="https://user-images.githubusercontent.com/63030539/235317165-b405ebc4-32f2-4eb9-b300-2f2d4689be23.png">


case 3: StrConv+Tr+BCE

<img width="524" alt="image" src="https://user-images.githubusercontent.com/63030539/235317184-2a3333f6-783b-4d2d-84d6-90219f6bddcb.png">


case 4: StrConv+Ups+Dice Loss

<img width="523" alt="image" src="https://user-images.githubusercontent.com/63030539/235317213-45755938-a106-4bc2-a4b8-4756d1067bd2.png">



from my observation in case 2 loss values are very less compared to remaining 3 cases. But in case 3 pet classified correctly one image chair classified correctly but in other cases chair classified as segmenatation (label class).



# part 2: VAE part

Misclassified images for MNIST data :

<img width="757" alt="image" src="https://user-images.githubusercontent.com/63030539/235317270-ea2b7bd5-ad82-4e6d-9bca-4fe5a22959f0.png">


