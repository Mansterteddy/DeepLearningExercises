import model

def main():
	net = model.Softmax()
	#net = model.DeepNN()
	net.train()

if __name__ == "__main__":
	main()
