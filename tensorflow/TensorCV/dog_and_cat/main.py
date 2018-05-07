import model

def main():
	net = model.VGGNet()
	#net = model.DeepNN()
	net.train()

if __name__ == "__main__":
	main()
