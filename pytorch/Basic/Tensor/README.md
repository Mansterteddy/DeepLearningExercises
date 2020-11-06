There are two things that pytorch Tensors have that numpy arrays lack:
1. pytorch Tensors can live on either GPU or CPU (numpy is cpu-only);
2. pytorch can automatically track tensor computations to enable automatic differentiation;
