# Atividade-com-GPU---05-08-2024
Programação Paralela Concorrente - Nesta atividade, configuramos um ambiente CUDA no Google Colab, compilamos e executamos um programa simples em GPU. Implementamos a soma de matrizes com kernels CUDA, perfilamos o desempenho e variamos o tamanho das matrizes para observar a variação nos tempos de execução.

    Abra o Colab e crie seu próprio notebook.

    Vá no menu Editar --> Configurações de Notebook

    Clique em Salvar

    Adicione uma célula de código para verificar a versão do CUDA.

    python

!nvidia-smi

Insira uma célula com os comandos abaixo para instalar uma extensão que permite compilar arquivos CUDA:

python

!pip install git+https://github.com/lesc-ufv/cad4u.git &> /dev/null
!git clone https://github.com/lesc-ufv/cad4u &> /dev/null
%load_ext plugin

Insira uma célula com o código abaixo:

python

%%writefile hellocuda.cu

#include <stdio.h>

__global__ void helloFromGPU()
{
    printf("Hello World from GPU! Block %d  Thread %d\n", blockIdx.x, threadIdx.x);
}

int main(int argc, char **argv)
{
    printf("Hello World from CPU!\n");

    helloFromGPU<<<1, 10>>>();
    cudaDeviceReset();

    cudaDeviceSynchronize();

    return 0;
}

Insira uma célula com os comandos para compilar e executar:

python

!nvcc hellocuda.cu -o hellocuda
!./hellocuda

Insira uma célula texto com o título "Atividade de soma de matrizes"

Insira uma célula de código e insira o código a seguir (%%gpu permite que o código seja compilado e executado ao executar a célula)

%%gpu
#include <stdio.h>
#include <assert.h>

#define N 4

__global__ void additionMatricesKernel(int *d_a, int *d_b, int *d_c){
  //Define os índices das threads
  int i = threadIdx.y+blockIdx.y*blockDim.y;
  int j = threadIdx.x+blockIdx.x*blockDim.x;
  //Realiza a soma de cada elemento
  d_c[i*N+j] = d_a[i*N+j] + d_b[i*N+j];
}

int main(){
  //Define as matrizes quadradas NxN
    int h_a[N][N], h_b[N][N], h_c[N][N];

  //Preenche as matrizes
    for(int i = 0; i < N; i++){
        for(int j = 0; j < N; j++){
            h_a[i][j] = h_b[i][j] = i+j;
            h_c[i][j] = 0;
        }
    }

    //Declara os ponteiros para a GPU
    int *d_a, *d_b, *d_c;

    const int nBytes = N * N * sizeof(int);

  //Aloca memória na GPU
    cudaMalloc((void**)&d_a,nBytes);
    cudaMalloc((void**)&d_b,nBytes);
    cudaMalloc((void**)&d_c,nBytes);

  //Copia dados da CPU para a GPU
    cudaMemcpy(d_a, h_a, nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_c, h_c, nBytes, cudaMemcpyHostToDevice);

  //Declaração do tamanho do grid e tamanho dos blocos
    dim3 blockSize(2,2);
    dim3 gridSize(N/2,N/2);

    //Executa o kernel
    additionMatricesKernel<<<gridSize,blockSize>>>(d_a, d_b, d_c);

  //Copia os dados de volta para a CPU
    (cudaMemcpy(h_c, d_c, nBytes, cudaMemcpyDeviceToHost));

  //Testa se a soma foi executada corretamente
    for(int i=0; i < N; i++){
        for(int j=0; j < N; j++){
            assert(h_a[i][j] + h_b[i][j] == h_c[i][j]);
        }
    }
  printf("\nSOMA EXECUTADA COM SUCESSO!\n\n");

  //Desaloca memória
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}

Explicação:

dim3 threadsBlocks(2,2) indica que as threads estarão dispostas em blocos que têm 2 dimensões e tamanho 2x2.
dim3 gridBlocks(N/Tx  x  N/Ty) indica que os blocos estarão dispostos num grid 2D de tamanho também N/Tx  x  N/Ty. No nosso exemplo Tx=Ty=2, portanto teremos N/2 x  N/2 Blocos

Além disso precisamos mapear os índices das threads aos índices dos elementos da matriz dentro do kernel, para isso definimos:

int i = threadIdx.y+blockIdx.y*blockDim.y
int j = threadIdx.x+blockIdx.x*blockDim.x

que são o número da thread em cada direção (x ou y) mais o número do bloco vezes o tamanho do bloco em cada direção. Dessa forma temos índices bidimensionais para localizar as threads no grid. Para mapeá-las no array unidimensional que representa a matriz basta fazer o índice do array igual a i*N+j, onde N é a dimensão da matriz. Por exemplo, uma matriz 4 x 4 pode ser mapeada a um array de 16 elementos (4 blocos 2 x 2), conforme figura a seguir.

Edite a célula de código e troque %%gpu por %%nvprof para exibir o profile da execução.


%%nvprof

Todo o código desta célula será executado e monitorado com o nvprof. Na sua opção padrão, a saída do código irá imprimir as mensagens do seu código seguidas das informações de medidas de tempo. Os resultados são impressos em uma lista ordenada com as funções que você executou seguido das API CUDA como cudaMalloc, etc. Para cada função serão exibidas as seguintes informações: a porcentagem do tempo total que a função utilizou, o tempo total, o número de vezes que a função foi chamada, seguido do tempo mínimo, médio e máximo. Na última coluna o nome da função.

Varie o valor de N para 8, 16, 32, 64 e observe a variação nos tempos de execução. A proporção entre o tempo de execução do kernel e os tempos de execução da função cudaMemCpy variou?
