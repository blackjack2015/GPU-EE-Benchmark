OBJ = o

ARCH = 70

NAME = fp32_fma

A1-FP32: $(NAME).$(OBJ)
	nvcc -arch=sm_$(ARCH) -o $@ $< -lcuda

$(NAME).$(OBJ): $(NAME).cu
	#nvcc -arch=sm_$(ARCH) -Xptxas -O0,-dlcm=cv,-dscm=wt  -c $(INCLUDES) $<
	nvcc -arch=sm_$(ARCH)  -c $(INCLUDES) $<

clean:
	rm -f $(NAME).$(OBJ) $(NAME) A1-FP32
