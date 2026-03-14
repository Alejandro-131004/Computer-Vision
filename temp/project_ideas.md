# Main Workflow

## Robustez

1. Antes de começar a parte 2, faz sentido ter uma filtragem dos intervalos de valores de intensidade para todos os parâmetros de cor HSV relativos às bolas (já segmentadas), para termos noção do tipo de cores possívieis.
2. Não podem existir duas bolas com o mesmo número (imagem deve ser flagged como "Suspicious" e identificar as ocorrências semelhantes)
3. Não como erro, apenas como flag para ver se a imagem foi corretamente segmentada: "ambas a bola branca e preta estão presentes?" (por acaso acontece nas 50 imagens que temos, mas nada garante que nas 10 de testes ou outras do dataset estará também, por isso não usar como restrição mas sim como apenas um check no desenvolvimento!)

## Questões Sobre o Evaluation Set

1. Podemos assumir que o dataset de teste tem sempre a bola branca e a bola preta? $\rightarrow$ **NÃO ASSUMIR ISSO!**
2. Podemos usar mais dados do mesmo dataset? Se sim, apenas para validação ou também no desenvolvimento? $\rightarrow$ podemos usar a vontade como validação e melhoria

## Part 1: Identificar Bolas por Shape 

1. Identificar mesa (blue edge detection?)
2. Primeiro fazer segmentação por shape (sempre circular, com uma threshold específica)
3. Bounding boxes quadradas em tudo o que apresenta círculo 
   1. Talvez seja boa ideia regular contrastes e sharpness para tornar essa detection mais robusta
   2. Shading? Sombras mais adaptadas 


## Part 2: Identificar Pontuação

1. Usar a pipeline da parte 1
2. Perform contrast+saturation balance em hsv (para termos mais controlo de saturação, brilho, ângulos de luz e etc)
   1. também é importante termos métodos de validação. Uma maneira de fazer isso é vermos a cor absoluta e os ratios de branco vs cor, mas também podemos ter um método relativo de diferença de cores, por exemplo: comparando esta bola à cor branca que já temos (por moda ou média de todas as imagens, isso é obtido na filtragem), quão vermelha é e etc. Isso teria um peso mínimo e essa weighted sum seria mais robusta que apenas cores absolutas (apenas uma ideia, não sei até que ponto é pertinente ou funciona, deixar para o fim!)
3. Na imagem de cor adaptada, identificar bounding boxes numericamente
   1. para distinguir entre bolas com as mesmas cobinações de cores, por exemplo a vermelha e branca, podemos ver o ratio de branco e vermelho nas bolas (pause) e só aí atribuir pontuação

## Part 3: Top View

1. Obter os cantos azuis novamente, e aplicar transformações para que os cantos tenham posições do género:
   1. linha reta entre canto superior e inferior
   2. linha reta entre canto esquerdo e direito
   3. canto azul na posição absoluta ??x?? da imagem gerada
