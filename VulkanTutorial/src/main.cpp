#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>

#include <iostream>
#include <vector>
#include <optional>
#include <set>
#include <algorithm>
#include <fstream>
#include <array>

// Window sizes
const uint32_t WIDTH = 800;
const uint32_t HEIGHT = 600;

// Maximum frames in flight
const int MAX_FRAMES_IN_FLIGHT = 2;

// Validation layers allow the program to be error checked during runtime
// Here we can create a vector and specify the Vulkan SDK built-in layers that we want
const std::vector<const char*> validationLayers = {
	"VK_LAYER_KHRONOS_validation" // All the useful standard validation
};

// Unlike the instance, these are extensions specific to the device
const std::vector<const char*> deviceExtensions = {
	VK_KHR_SWAPCHAIN_EXTENSION_NAME // Enables the swap chain
};

// We can enable/disable layers depending on whether the program is in debug/release
#ifdef NDEBUG // Part of the C++ standard, means "not debug"
const bool enableValidationLayers = false;
#else
const bool enableValidationLayers = true;
#endif

VkResult CreateDebugUtilsMessengerEXT(VkInstance instance, const VkDebugUtilsMessengerCreateInfoEXT* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkDebugUtilsMessengerEXT* pDebugMessenger) {
	/*
	* Like the Vulkan instance, the struct in setupDebugMessenger needs to be passed to a function to create the debug messenger object.
	* Although, since this function is from an extension, it is not automatically loaded.
	* We must use a function called vkGetInstanceProcAddr to look up its address.
	*/
	auto func = (PFN_vkCreateDebugUtilsMessengerEXT)vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT");
	if (func != nullptr) { // vkGetInstanceProcAddr will return a nullptr if the function is not found 
		return func(instance, pCreateInfo, pAllocator, pDebugMessenger); // Calls the function, remember the return value is of type VkResult
	} else {
		return VK_ERROR_EXTENSION_NOT_PRESENT; // If nullptr is returned, extension must not be present
	}
}

void DestroyDebugUtilsMessengerEXT(VkInstance instance, VkDebugUtilsMessengerEXT debugMessenger, const VkAllocationCallbacks* pAllocator) {
	/*
	* Just like the Vulkan instance, the debug messenger object must be cleaned up.
	* Again, since it's from an extension, the function needs to be loaded with vkGetInstanceProcAddr.
	*/
	auto func = (PFN_vkDestroyDebugUtilsMessengerEXT)vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT");
	if (func != nullptr) {
		func(instance, debugMessenger, pAllocator);
	}
}

struct QueueFamilyIndices {
	/*
	* This struct indicates the queue families we want to use.
	* Not all graphics cards may support these queue families, which is why they are wrapped in optional.
	* This allows properties to be checked if it has a value or not, therefore they are allowed to have no value.
	*/
	std::optional<uint32_t> graphicsFamily; // Drawing commands
	std::optional<uint32_t> presentFamily; // Presents rendered images to a surface

	bool isComplete() {
		// Checks that all required properties in the struct have a value, aka the queue families are supported by the graphics card
		return graphicsFamily.has_value() && presentFamily.has_value();
	}
};

struct SwapChainSupportDetails {
	/*
	* This struct holds details about our swap chain.
	* It includes properties that we need to check to ensure the swap chain works for our use.
	* The struct has a similar use to QueueFamilyIndices.
	*/
	VkSurfaceCapabilitiesKHR capabilities; // E.g. min/max images in chain, min/max width and height of images
	std::vector<VkSurfaceFormatKHR> formats; // E.g. pixel format, colour space
	std::vector<VkPresentModeKHR> presentModes; // Available presentation modes, e.g. vertical sync, triple buffering
};

struct Vertex {
	/*
	* Represents a single vertex consisting of its position and colour.
	*/
	glm::vec2 pos;
	glm::vec3 colour;

	static VkVertexInputBindingDescription getBindingDescription() {
		/*
		* Describes the rate to load data from memory throughout the vertices.
		*/
		VkVertexInputBindingDescription bindingDescription{};
		bindingDescription.binding = 0; // Index of the binding in the array of bindings
		bindingDescription.stride = sizeof(Vertex); // Number of bytes from one entry to the next
		bindingDescription.inputRate = VK_VERTEX_INPUT_RATE_VERTEX; // Move to next data entry after each vertex (not each instance)

		return bindingDescription;
	}

	static std::array<VkVertexInputAttributeDescription, 2> getAttributeDescriptions() {
		/*
		* Describes how to extract a vertex attribute from a chunk of vertex data originating from a binding description.
		* Each vertex has two attributes, position and colour, so we need two attribute description structures.
		*/
		std::array<VkVertexInputAttributeDescription, 2> attributeDescriptions{};

		attributeDescriptions[0].binding = 0; // Binding the per-vertex data comes from
		attributeDescriptions[0].location = 0; // References the location directive of the input in the vertex shader (position is at location = 0)
		attributeDescriptions[0].format = VK_FORMAT_R32G32_SFLOAT; // Type of data for the attribute (vec2 for position)
		attributeDescriptions[0].offset = offsetof(Vertex, pos); // Number of bytes since the start of the per-vertex data to read from

		attributeDescriptions[1].binding = 0; // Both attributes come from same binding
		attributeDescriptions[1].location = 1; // Colour attribute is at location = 1
		attributeDescriptions[1].format = VK_FORMAT_R32G32B32_SFLOAT; // Type of data is vec3 for colour
		attributeDescriptions[1].offset = offsetof(Vertex, colour); // Reads from where the colour data is stored per-vertex

		return attributeDescriptions;
	}
};

// Vertex data to represent a rectangle
const std::vector<Vertex> vertices = {
	{{-0.5f, -0.5f}, {1.0f, 0.0f, 0.0f}},
	{{0.5f, -0.5f}, {0.0f, 1.0f, 0.0f}},
	{{0.5f, 0.5f}, {0.0f, 0.0f, 1.0f}},
	{{-0.5f, 0.5f}, {1.0f, 1.0f, 1.0f}}
};

// Index data to reuse multiple vertices in the above data
const std::vector<uint16_t> indices = {
	0, 1, 2, 2, 3, 0
};

class HelloTriangleApplication {
public:
	void run() {
		initWindow(); // Initialise the GLFW window

		initVulkan(); // Initialise Vulkan

		mainLoop(); // Run the main loop

		cleanup(); // Cleanup objects
	}

private:
	GLFWwindow* window; // GLFW window

	VkInstance instance; // The connection between the application and Vulkan
	VkDebugUtilsMessengerEXT debugMessenger; // Responsible for handling debug callbacks
	VkSurfaceKHR surface; // A surface for Vulkan to present rendered images to, backed by the GLFW window

	VkPhysicalDevice physicalDevice = VK_NULL_HANDLE; // The graphics card Vulkan will be using
	VkDevice device; // The logical device Vulkan will be using

	VkQueue graphicsQueue; // Graphics queue handle
	VkQueue presentQueue; // Presentation queue handle

	VkSwapchainKHR swapChain; // Swap chain used to render images
	std::vector<VkImage> swapChainImages; // Handles for the images in the swap chain
	VkFormat swapChainImageFormat; // Format of the swap chain images
	VkExtent2D swapChainExtent; // Extent of the swap chain images
	std::vector<VkImageView> swapChainImageViews; // Holds the image views for each image in the swap chain
	std::vector<VkFramebuffer> swapChainFramebuffers; // Each framebuffer corresponds to an image in the swap chain

	VkRenderPass renderPass; // Render pass to be used by the graphics pipeline
	VkPipelineLayout pipelineLayout; // Specifies uniform values for shaders
	VkPipeline graphicsPipeline; // Graphics pipeline

	VkCommandPool commandPool; // Manages memory used to store buffers and command buffers are allocated from them

	VkBuffer vertexBuffer; // Buffer that stores vertex data
	VkDeviceMemory vertexBufferMemory; // Stores the vertex buffer memory
	VkBuffer indexBuffer; // Buffer that stores index data
	VkDeviceMemory indexBufferMemory; // Stores the index buffer memory

	std::vector<VkCommandBuffer> commandBuffers; // Records command operations to be performed on the GPU (freed with command pool so cleanup is not needed)

	std::vector<VkSemaphore> imageAvailableSemaphores; // Signals when a image is acquired from the swapchain and is ready for rendering
	std::vector<VkSemaphore> renderFinishedSemaphores; // Signals when that rendering is finished and presentation can happen
	std::vector<VkFence> inFlightFences; // Signal to ensure only one frame is rendering at a time
	uint32_t currentFrame = 0; // Current frame being processed

	bool framebufferResized = false; // Indicates whether a window resize has occured

	void initWindow() {
		/*
		* Initialises the GLFW window.
		*/
		glfwInit(); // This must be called to initialise the GLFW library

		// glfwWindowHint sets options for the window
		glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API); // Tells GLFW to not create an OpenGL context, since that's what it was first built to do

		window = glfwCreateWindow(WIDTH, HEIGHT, "Vulkan", nullptr, nullptr); // Creates a GLFW window, simple enough

		glfwSetWindowUserPointer(window, this); // Sets an arbitrary pointer of the application object for the specified window
		glfwSetFramebufferSizeCallback(window, framebufferResizeCallback); // Sets the function to be called when the window is resized
	}

	static void framebufferResizeCallback(GLFWwindow* window, int width, int height) {
		/*
		* Called whenever the window is resized.
		*/
		auto app = reinterpret_cast<HelloTriangleApplication*>(glfwGetWindowUserPointer(window)); // Get the application object using the window
		app->framebufferResized = true; // Tells the application that the window has been resized
	}

	void initVulkan() {
		/*
		* Initalises Vulkan and its objects.
		*/
		createInstance(); // Creates an instance of Vulkan

		setupDebugMessenger(); // Sets up a debug messenger

		createSurface(); // Creates the surface for Vulkan to present rendered images

		pickPhysicalDevice(); // Chooses a graphics card to use

		createLogicalDevice(); // Creates the logical device

		createSwapChain(); // Creates the swap chain

		createImageViews(); // Creates the image views for each image in the swap chain

		createRenderPass(); // Creates the render pass for the graphics pipeline

		createGraphicsPipeline(); // Creates the graphics pipeline

		createFramebuffers(); // Creates a framebuffer for each image in the swap chain

		createCommandPool(); // Creates the command pool

		createVertexBuffer(); // Creates a vertex buffer

		createIndexBuffer(); // Creates an index buffer

		createCommandBuffers(); // Allocates the command buffers

		createSyncObjects(); // Creates the semaphores and fences
	}

	void mainLoop() {
		/*
		* Main loop for the application.
		*/
		// Runs as long as the window is open
		while (!glfwWindowShouldClose(window)) {
			glfwPollEvents(); // Checks for input events
			drawFrame(); // Draw a frame
		}

		vkDeviceWaitIdle(device); // Waits for drawing and presentation operations to finish
	}

	void cleanupSwapChain() {
		/*
		* Cleans up the swap chain and its objects.
		*/
		// Destroys each framebuffer
		for (VkFramebuffer framebuffer : swapChainFramebuffers) {
			vkDestroyFramebuffer(device, framebuffer, nullptr);
		}

		// Destroys each image view
		for (VkImageView imageView : swapChainImageViews) {
			vkDestroyImageView(device, imageView, nullptr);
		}

		vkDestroySwapchainKHR(device, swapChain, nullptr); // Destroys the swap chain
	}

	void cleanup() {
		/*
		* Cleans up all objects in the application.
		*/
		cleanupSwapChain(); // Destroys the swap chain and all its dependable objects

		vkDestroyBuffer(device, indexBuffer, nullptr); // Destroys the index buffer
		vkFreeMemory(device, indexBufferMemory, nullptr); // Frees the index buffer memory

		vkDestroyBuffer(device, vertexBuffer, nullptr); // Destroys the vertex buffer
		vkFreeMemory(device, vertexBufferMemory, nullptr); // Frees the vertex buffer memory

		vkDestroyPipeline(device, graphicsPipeline, nullptr); // Destroys the graphics pipeline
		vkDestroyPipelineLayout(device, pipelineLayout, nullptr); // Destroys the pipeline layout

		vkDestroyRenderPass(device, renderPass, nullptr); // Destroys the render pass

		// Destroys semaphores and fences for each frame in flight
		for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
			vkDestroySemaphore(device, imageAvailableSemaphores[i], nullptr); // Destroys the image available semaphore
			vkDestroySemaphore(device, renderFinishedSemaphores[i], nullptr); // Destroys the render finished semaphore
			vkDestroyFence(device, inFlightFences[i], nullptr); // Destroys the frame in flight fence
		}

		vkDestroyCommandPool(device, commandPool, nullptr); // Destroys the command pool

		vkDestroyDevice(device, nullptr); // Destroys the logical device

		// Destroys the Vulkan debug messenger if validation layers are enabled
		if (enableValidationLayers) {
			DestroyDebugUtilsMessengerEXT(instance, debugMessenger, nullptr);
		}

		vkDestroySurfaceKHR(instance, surface, nullptr); // Destroys the Vulkan surface
		vkDestroyInstance(instance, nullptr); // Destroys the Vulkan instance

		glfwDestroyWindow(window); // Destroys the window and its context

		glfwTerminate(); // Terminates the GLFW library
	}

	void recreateSwapChain() {
		/*
		* Recreates the swap chain and all objects that depend on it or the window size.
		*/
		// If window is minimised, pause it until it is back in the foreground
		int width = 0, height = 0;
		glfwGetFramebufferSize(window, &width, &height);
		while (width == 0 || height == 0) {
			glfwGetFramebufferSize(window, &width, &height);
			glfwWaitEvents(); // Pauses the window
		}

		vkDeviceWaitIdle(device); // Wait for other resources to finish before accessing

		cleanupSwapChain(); // Destroy old swap chain objects

		createSwapChain(); // Recreate swap chain again
		createImageViews(); // Recreate image views
		createFramebuffers(); // Recreate frame buffers
	}

	void createInstance() {
		/*
		* In order to create an instance, some structs must be filled out first.
		* The compulsory struct is VkInstanceCreateInfo, and VkApplicationInfo is optional, but recommended.
		* Once VkInstanceCreateInfo is created, it is then provided to vkCreateInstance to create an instance for Vulkan.
		* This is generally how objects in Vulkan are created.
		*/
		// Before any Vulkan code is run, the requested validation layers should be checked for availability
		if (enableValidationLayers && !checkValidationLayerSupport()) {
			throw std::runtime_error("Validation layers requested, but not available!");
		}

		VkApplicationInfo appInfo{}; // A struct to hold information about the application
		// While optional, it essentially provides the driver with useful information to optimise the application
		appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO; // Many structs for Vulkan require you to explicitly specify the type
		appInfo.pApplicationName = "Hello Triangle";
		appInfo.applicationVersion = VK_MAKE_API_VERSION(0, 1, 0, 0);
		appInfo.pEngineName = "No Engine";
		appInfo.engineVersion = VK_MAKE_API_VERSION(0, 1, 0, 0);
		appInfo.apiVersion = VK_API_VERSION_1_0;

		VkInstanceCreateInfo createInfo{}; // Compulsory struct that tells the Vulkan driver which global extensions and validation layers to use
		createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
		createInfo.pApplicationInfo = &appInfo;

		std::vector<const char*> extensions = getRequiredExtensions(); // Obtains the required extensions for the program
		createInfo.enabledExtensionCount = static_cast<uint32_t>(extensions.size()); // Amount of enabled extensions
		createInfo.ppEnabledExtensionNames = extensions.data(); // Enabled extensions by name

		// If validation layers are enabled, include them in the instance create info properties and create another debug messenger
		VkDebugUtilsMessengerCreateInfoEXT debugCreateInfo{}; // This is a seperate debug messenger created specifically for debugging vkCreateInstance and vkDestroyInstance
		if (enableValidationLayers) {
			createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size()); // Amount of enabled layers
			createInfo.ppEnabledLayerNames = validationLayers.data(); // Enabled layers by name

			populateDebugMessengerCreateInfo(debugCreateInfo);
			createInfo.pNext = (VkDebugUtilsMessengerCreateInfoEXT*)&debugCreateInfo; // The debug messenger can be placed in this field of createInfo
			// It also allows the object to be destroyed automatically when the Vulkan instance is
		} else {
			createInfo.enabledLayerCount = 0; // Zero layers if disabled

			createInfo.pNext = nullptr;
		}

		// Creates an instance for Vulkan, and checks for failure in which case it will provide the error
		// Almost all Vulkan functions return a value of type VkResult, meaning they can all be tested in this way
		if (vkCreateInstance(&createInfo, nullptr, &instance) != VK_SUCCESS) {
			throw std::runtime_error("Failed to create instance!");
		}
	}

	void populateDebugMessengerCreateInfo(VkDebugUtilsMessengerCreateInfoEXT& createInfo) {
		/*
		* Populates the debug messenger create info struct with the settings we need.
		*/
		createInfo = {};
		createInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT; // Remember to specify the type of struct it is
		createInfo.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
		createInfo.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
		createInfo.pfnUserCallback = debugCallback; // More info about the above parameters can be found in this function
	}

	void setupDebugMessenger() {
		/*
		* Sets up a debug messenger for Vulkan if validation layers are enabled.
		* In a similar way to the Vulkan instance, a create info struct is needed to provide details about the object.
		*/
		if (!enableValidationLayers) return; // Only run if validation layers are enabled

		// Just like the Vulkan instance, other objects like this are setup through structs
		VkDebugUtilsMessengerCreateInfoEXT createInfo{};
		populateDebugMessengerCreateInfo(createInfo);

		// Creates the debug messenger object
		// A proxy function we created is used to find the function that creates the object, since its from an extension
		if (CreateDebugUtilsMessengerEXT(instance, &createInfo, nullptr, &debugMessenger) != VK_SUCCESS) {
			throw std::runtime_error("Failed to set up debug messenger!");
		}
	}

	void createSurface() {
		/*
		* Creates a surface for Vulkan to present rendered images to.
		* Normally, there would need to be platform-specific code to setup up a surface for each platform we want to support.
		* But GLFW supports many platforms, so it's easier for us to use this function that does all of the hard work for us.
		*/
		if (glfwCreateWindowSurface(instance, window, nullptr, &surface) != VK_SUCCESS) {
			throw std::runtime_error("Failed to create window surface");
		}
	}

	void pickPhysicalDevice() {
		/*
		* The physical device is the graphics card that Vulkan will be using.
		* We can list the graphics cards available just like with extensions and validation layers.
		* Then we need to check if the device is suitable for our uses, and if it is, it can be used.
		*/
		uint32_t deviceCount = 0;
		vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr); // Finds all available devices

		// If zero supported devices are found, throw an error
		if (deviceCount == 0) {
			throw std::runtime_error("Failed to find GPUs with Vulkan support!");
		}

		std::vector<VkPhysicalDevice> devices(deviceCount); // Create an array to hold all supported devices
		vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data());

		// Loops through the supported physical devices to find one that is suitable for our uses
		for (const VkPhysicalDevice& device : devices) {
			if (isDeviceSuitable(device)) {
				physicalDevice = device;
				break;
			}
		}

		// If no suitable device is found, throw an error
		if (physicalDevice == VK_NULL_HANDLE) {
			throw std::runtime_error("Failed to find a suitable GPU!");
		}
	}

	void createLogicalDevice() {
		/*
		* Creates the logical device, which essentially links the physical device to Vulkan so commands can be run through it.
		* The first task is to specify the queues Vulkan will be using from the queue families we requested.
		* This is also where device features can be enabled for use.
		* Lastly, much like the Vulkan instance object, it requires a main create info struct to be filled in.
		* Devices can also use extensions and validation layers, but unlike the Vulkan instance, these are physical device specific.
		* Validation layers for the device are ignored by up-to-date implementations, since the Vulkan instance handles these anyways.
		* Although, it is still recommended to specify these to be compatible with older implementations.
		*/
		QueueFamilyIndices indices = findQueueFamilies(physicalDevice); // Accesses the queue families for the physical device

		std::vector<VkDeviceQueueCreateInfo> queueCreateInfos; // Because we will be making a queue for each family
		std::set<uint32_t> uniqueQueueFamilies = { indices.graphicsFamily.value(), indices.presentFamily.value() }; // Holds each queue family index

		// Since multiple queue families are required, we loop through each one defining their create infos
		float queuePriority = 1.0f;
		for (uint32_t queueFamily : uniqueQueueFamilies) {
			VkDeviceQueueCreateInfo queueCreateInfo{}; // Describes the queue we are creating
			queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
			queueCreateInfo.queueFamilyIndex = queueFamily; // Index of the queue family
			queueCreateInfo.queueCount = 1; // Only creating one queue per queue family
			queueCreateInfo.pQueuePriorities = &queuePriority; // Influences the scheduling of command buffer execution (required even for one queue)
			queueCreateInfos.push_back(queueCreateInfo); // Add the queue to the list
		}

		VkPhysicalDeviceFeatures deviceFeatures{}; // Specifies the set of device features we want to use (leaving this blank for now)

		// Specifies the create info for the logical device, it has similar fields to the Vulkan instance one
		VkDeviceCreateInfo createInfo{}; // The main device create info struct
		createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;

		createInfo.queueCreateInfoCount = static_cast<uint32_t>(queueCreateInfos.size()); // Amount of queue create infos
		createInfo.pQueueCreateInfos = queueCreateInfos.data(); // Points to the queue create info structs

		createInfo.pEnabledFeatures = &deviceFeatures; // Points to the enabled device features

		createInfo.enabledExtensionCount = static_cast<uint32_t>(deviceExtensions.size()); // Amount of device extensions
		createInfo.ppEnabledExtensionNames = deviceExtensions.data(); // List of enabled device extensions

		// (Optional) These fields are ignored in up-to-date implementations of Vulkan,
		// but it's still recommended to specify these anyways to be compatible with older implementations
		if (enableValidationLayers) {
			createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
			createInfo.ppEnabledLayerNames = validationLayers.data();
		} else {
			createInfo.enabledLayerCount = 0;
		}

		// The logical device can now be created
		if (vkCreateDevice(physicalDevice, &createInfo, nullptr, &device) != VK_SUCCESS) {
			throw std::runtime_error("Failed to create logical device!");
		}

		// The queues are automatically created with the logical device, but this creates handles to interface with them
		vkGetDeviceQueue(device, indices.graphicsFamily.value(), 0, &graphicsQueue);
		vkGetDeviceQueue(device, indices.presentFamily.value(), 0, &presentQueue);
	}

	void createSwapChain() {
		/*
		* Creates the swap chain for Vulkan.
		* The swap chain is a queue of rendered images ready to be presented to the screen.
		* Firstly, we queury the selected physical device to choose settings that best suit our needs.
		* Then we choose the minimum amount of images we want in the swap chain, ensuring it doesn't exceed the maximum amount.
		* Just like other Vulkan objects, a struct is then created holding the create info of the swap chain we must fill in.
		* Properties from the queue families must be queuried as well, so that images are handled correctly across them.
		* A few more fields are filled in, then the swap chain can be created using the usual Vulkan object creation function.
		* The swap chain image handles can then be retrieved.
		*/
		SwapChainSupportDetails swapChainSupport = queurySwapChainSupport(physicalDevice); // Gets swap chain support details for the selected physical device

		VkSurfaceFormatKHR surfaceFormat = chooseSwapSurfaceFormat(swapChainSupport.formats); // Choose a surface format
		VkPresentModeKHR presentMode = chooseSwapPresentMode(swapChainSupport.presentModes); // Choose a swap present mode
		VkExtent2D extent = chooseSwapExtent(swapChainSupport.capabilities); // Choose a swap extent

		// Minimum amount of images in the swap chain
		uint32_t imageCount = swapChainSupport.capabilities.minImageCount + 1; // Sometimes we have to wait for the driver to complete internal operations between images, hence +1

		// Ensure not to exceed the maximum image count after calculating the above
		if (swapChainSupport.capabilities.maxImageCount > 0 && imageCount > swapChainSupport.capabilities.maxImageCount) { // 0 means there is no max
			imageCount = swapChainSupport.capabilities.maxImageCount;
		}

		VkSwapchainCreateInfoKHR createInfo{};
		createInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
		createInfo.surface = surface; // Surface to render images from the swap chain to

		createInfo.minImageCount = imageCount; // Minimum amount of images in the swap chain
		createInfo.imageFormat = surfaceFormat.format; // Colour format of the images
		createInfo.imageColorSpace = surfaceFormat.colorSpace; // Colour space of the images
		createInfo.imageExtent = extent; // Pixel size of the images (should be same as window size)
		createInfo.imageArrayLayers = 1; // Amount of layers each image consists of, always one (>1 is used for stereoscopic 3D applications)
		createInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT; // Rendering directly to the images, meaning they're used as color attachment
		// Another option is to render images to seperate images first to perform operations, aka post-processing

		QueueFamilyIndices indices = findQueueFamilies(physicalDevice); // Gets the queue family indices obtained from the physical device
		uint32_t queueFamilyIndices[] = { indices.graphicsFamily.value(), indices.presentFamily.value() };

		// We need to check if the two queues are in the same family index, so that images are handled correctly between multiple queues
		if (indices.graphicsFamily != indices.presentFamily) {
			createInfo.imageSharingMode = VK_SHARING_MODE_CONCURRENT; // Means images can be used across multiple queue families
			createInfo.queueFamilyIndexCount = 2; // Amount of queues concurrently used
			createInfo.pQueueFamilyIndices = queueFamilyIndices; // The queues that images will be sharing concurrent use
		} else {
			createInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE; // Means images need to be explicitly transferred between queue families
			createInfo.queueFamilyIndexCount = 0; // Optional
			createInfo.pQueueFamilyIndices = nullptr; // Optional
		}

		createInfo.preTransform = swapChainSupport.capabilities.currentTransform; // Specifies any transformations to images in the swap chain if supported (set to none)
		createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR; // Specifies if the alpha channel should be used for blending with other windows (set to ignore)
		createInfo.presentMode = presentMode; // Specifies the presentation mode of the swap chain
		createInfo.clipped = VK_TRUE; // Specifies whether we care about the colour of pixels that are obscured, e.g. if a window is in front of them (true means we don't care)

		// It's possible for the swap chain to become invalid or unoptimized, e.g. when the window is resized
		// In this case a new swap chain needs to be created and a reference to the old is specified in this field
		// We will handle this another time, for now lets assume there will only be one swap chain
		createInfo.oldSwapchain = VK_NULL_HANDLE;

		// With all that done, we can call the usual Vulkan object creation function to create the swap chain
		if (vkCreateSwapchainKHR(device, &createInfo, nullptr, &swapChain) != VK_SUCCESS) {
			throw std::runtime_error("Failed to create swap chain!");
		}

		// The swap chain images can now be obtained in the same way any other list of Vulkan objects are
		vkGetSwapchainImagesKHR(device, swapChain, &imageCount, nullptr); // Queuries the final number of swap chain images
		swapChainImages.resize(imageCount);
		vkGetSwapchainImagesKHR(device, swapChain, &imageCount, swapChainImages.data()); // Retrieve the image handles

		// Lastly, store the swap chain image format and extent in member variables
		swapChainImageFormat = surfaceFormat.format;
		swapChainExtent = extent;
	}

	void createImageViews() {
		/*
		* Creates the image views for each image in the swap chain.
		* Image views describe how to access the image and which part of the image to access.
		* E.g. if it should be treated as a 2D texture depth texture without any mipmapping levels.
		*/
		swapChainImageViews.resize(swapChainImages.size()); // Resize the list to fit all the image views

		// Loops through each image in the swap chain to create a corresponding image view
		for (size_t i = 0; i < swapChainImages.size(); i++) {
			VkImageViewCreateInfo createInfo{}; // Struct to hold image view creation data
			createInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
			createInfo.image = swapChainImages[i]; // Image to create image view for

			createInfo.viewType = VK_IMAGE_VIEW_TYPE_2D; // Image can also be 1D or 3D
			createInfo.format = swapChainImageFormat; // Format of the image

			// The colour channels can be mixed around to produce different effects (e.g. a monochrome texture when all channels are red)
			createInfo.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
			createInfo.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
			createInfo.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
			createInfo.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;

			// Describes what the image's purpose is and which part of the image should be accessed
			createInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT; // Images will be used as colour targets
			createInfo.subresourceRange.baseMipLevel = 0; // No mipmapping levels
			createInfo.subresourceRange.levelCount = 1;
			createInfo.subresourceRange.baseArrayLayer = 0; // No multiple layers (a stereographic 3D application could be created with multiple layers)
			createInfo.subresourceRange.layerCount = 1;

			// Now the image view can be created
			if (vkCreateImageView(device, &createInfo, nullptr, &swapChainImageViews[i]) != VK_SUCCESS) {
				throw std::runtime_error("Failed to create image views!");
			}
		}
	}

	void createRenderPass() {
		/*
		* Creates the render pass for the graphics pipeline.
		* Render passes consist of attachments, attachment references, subpasses, and subpass dependencies.
		*/
		// In this case we'll use a single colour buffer attachment represented by one image of the swap chain
		VkAttachmentDescription colourAttachment{};
		colourAttachment.format = swapChainImageFormat; // Matches swap chain image format
		colourAttachment.samples = VK_SAMPLE_COUNT_1_BIT; // No multisampling so 1 bit

		// These fields determine what to do with the data before and after rendering
		colourAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR; // Means framebuffer will be cleared to black before rendering again
		colourAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE; // Rendered contents will be stored in memory to be used later

		// These fields only apply to applications using stencil data, which this one does not
		colourAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
		colourAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;

		// Images need to be transitioned to specific layouts depending on what operations they will be involved in next
		colourAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED; // Layout of image before render pass begins
		colourAttachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR; // Layout to transition to after render pass finishes

		// This structure holds information to reference an attachment previously described, in this case the colour attachment
		VkAttachmentReference colourAttachmentRef{};
		colourAttachmentRef.attachment = 0; // Which attachment to reference by index in the array
		colourAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL; // Layout to transition to when the subpass starts

		// Description about a subpass consisting of attachments
		VkSubpassDescription subpass{};
		subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS; // This subpass is a graphics subpass
		subpass.colorAttachmentCount = 1; // Number of attachments in subpass
		subpass.pColorAttachments = &colourAttachmentRef; // References to the attachments for use
		// The index of the attachment in this array is directly referenced from shaders with layout(location = 0)

		// Specifies memory and execution dependencies between subpasses
		VkSubpassDependency dependency{};
		dependency.srcSubpass = VK_SUBPASS_EXTERNAL; // Refers to implicit subpass before or after render pass depending on if it is specified in srcSubpass or dstSubpass
		dependency.dstSubpass = 0; // Refers to the index of the subpass (must be higher than srcSubpass to prevent cycles in dependency graph)
		dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT; // Operation to wait on
		dependency.srcAccessMask = 0;
		dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT; // Operation that should wait on this are in colour attachment stage,
		dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT; // and involve wiritng the colour attachment

		// With all attachments and subpasses created, the render pass create info can now be filled in
		VkRenderPassCreateInfo renderPassInfo{};
		renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
		renderPassInfo.attachmentCount = 1;
		renderPassInfo.pAttachments = &colourAttachment;
		renderPassInfo.subpassCount = 1;
		renderPassInfo.pSubpasses = &subpass;
		renderPassInfo.dependencyCount = 1;
		renderPassInfo.pDependencies = &dependency;

		// Render pass can now be created
		if (vkCreateRenderPass(device, &renderPassInfo, nullptr, &renderPass) != VK_SUCCESS) {
			throw std::runtime_error("Failed to create render pass!");
		}
	}

	void createGraphicsPipeline() {
		/*
		* Creates a graphics pipeline.
		* Once a pipeline is created, it cannot be changed without building another pipeline.
		* There are many steps involved in creating a pipeline, as shown below.
		* Shader Modules: Shader code is loaded in as shader modules, which act as thin wrappers around the shader bytecode.
		* Vertex Input & Input Assembly: The actual vertex data needs to be loaded in as well as specifying how it is structured.
		* Viewports & Scissors: Dynamic states need to be defined, which in this case are the viewport and scissor states.
		* Rasterizer: Configuring the rasterizer is required so geometry from the vertex shader can be turned into fragments.
		* Multisampling: This is a way to perform anti-aliasing and requires a GPU feature to be enabled.
		* Colour Blending: Blends each fragments colour with a colour that is already in the framebuffer.
		* Pipeline Layout: Specifies uniform values for use in shaders.
		* Render Pass: Not described in this function although is required.
		* After all these structures are defined, the graphics pipeline can be created.
		*/
		// Reads the shader code from each file
		std::vector<char> vertShaderCode = readFile("shaders/vert.spv");
		std::vector<char> fragShaderCode = readFile("shaders/frag.spv");

		// Creates the shader modules using the shader code obtained from above
		VkShaderModule vertShaderModule = createShaderModule(vertShaderCode);
		VkShaderModule fragShaderModule = createShaderModule(fragShaderCode);

		// Vertex shader stage create info
		VkPipelineShaderStageCreateInfo vertShaderStageInfo{};
		vertShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		vertShaderStageInfo.stage = VK_SHADER_STAGE_VERTEX_BIT; // Stage the shader is going to be used
		vertShaderStageInfo.module = vertShaderModule; // Shader module containing the code for this stage
		vertShaderStageInfo.pName = "main"; // Specifies the entry point for the shader code
		// This allows multiple shaders to be used in the same module, each with a different entry point

		// Fragment shader stage create info (same fields as above)
		VkPipelineShaderStageCreateInfo fragShaderStageInfo{};
		fragShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		fragShaderStageInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
		fragShaderStageInfo.module = fragShaderModule;
		fragShaderStageInfo.pName = "main";

		// An array to reference the above shader stages during pipeline creation
		VkPipelineShaderStageCreateInfo shaderStages[] = { vertShaderStageInfo, fragShaderStageInfo };

		// This structure describes the format of the vertex data to be passed to the vertex shader
		VkPipelineVertexInputStateCreateInfo vertexInputInfo{};
		vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;

		auto bindingDescription = Vertex::getBindingDescription();
		auto attributeDescriptions = Vertex::getAttributeDescriptions();

		vertexInputInfo.vertexBindingDescriptionCount = 1;
		vertexInputInfo.pVertexBindingDescriptions = &bindingDescription; // Bindings is the spacing between the data and whether its per-vertex or per-instance
		vertexInputInfo.vertexAttributeDescriptionCount = static_cast<uint32_t>(attributeDescriptions.size());
		vertexInputInfo.pVertexAttributeDescriptions = attributeDescriptions.data(); // Type of attributes passed to vertex shader, which binding to load from and which offset

		// This structure describes what kind of geometry will be drawn from the vertices
		VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
		inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
		inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST; // Triangle from every three vertices without reuse
		inputAssembly.primitiveRestartEnable = VK_FALSE; // When true can be used with _STRIP topology modes

		// Dynamic states can be changed even after the pipeline has been created, and are specified below
		std::vector<VkDynamicState> dynamicStates = {
			VK_DYNAMIC_STATE_VIEWPORT, // Consists of where the image is drawn to, almost always the swap chain extent (aka transformation)
			VK_DYNAMIC_STATE_SCISSOR // Acts as a filter for which region of pixels to draw to the screen (aka cropping)
		};

		// This struct holds data about the pipelines dynamic states
		VkPipelineDynamicStateCreateInfo dynamicState{};
		dynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
		dynamicState.dynamicStateCount = static_cast<uint32_t>(dynamicStates.size()); // Number of dynamic states
		dynamicState.pDynamicStates = dynamicStates.data(); // Dynamic states

		// If the viewport and scissor states are set to dynamic, we only need to specify their count
		VkPipelineViewportStateCreateInfo viewportState{};
		viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
		viewportState.viewportCount = 1;
		viewportState.pViewports = nullptr;
		viewportState.scissorCount = 1;
		viewportState.pScissors = nullptr;

		// Rasterization takes geometry from the vertex shader and turns it into fragments for the fragment shader
		// A few options in this structure requires enabling GPU features
		VkPipelineRasterizationStateCreateInfo rasterizer{};
		rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
		rasterizer.depthClampEnable = VK_FALSE; // When true, fragments beyond near and far planes are clamped instead of discarded
		rasterizer.rasterizerDiscardEnable = VK_FALSE; // When true, geometry never passes through the rasterizer stage (disables output to framebuffer)
		rasterizer.polygonMode = VK_POLYGON_MODE_FILL; // Determines how fragments are generated for geometry
		rasterizer.lineWidth = 1.0f; // Describes the thickness of lines in terms of number of fragments
		rasterizer.cullMode = VK_CULL_MODE_BACK_BIT; // Determines type of face culling to use
		rasterizer.frontFace = VK_FRONT_FACE_CLOCKWISE; // Specifies the vertex order for faces to be considered front-facing
		rasterizer.depthBiasEnable = VK_FALSE; // When true, can then alter depth values (other create info fields)

		// Multisampling is one of the ways to perform anti-aliasing
		VkPipelineMultisampleStateCreateInfo multisampling{};
		multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
		multisampling.sampleShadingEnable = VK_FALSE; // Multisampling is disabled for now
		multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

		// Contains the configuration per attached framebuffer
		VkPipelineColorBlendAttachmentState colourBlendAttachment{};
		colourBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
		colourBlendAttachment.blendEnable = VK_FALSE; // Colour blending is disabled for now

		// Contains the global colour blending settings
		VkPipelineColorBlendStateCreateInfo colourBlending{};
		colourBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
		colourBlending.logicOpEnable = VK_FALSE; // When true, perfoms blending using bitwise combination
		colourBlending.attachmentCount = 1; // Amount of attachments
		colourBlending.pAttachments = &colourBlendAttachment; // Colour blend attachments create info

		// Uniform values can be used for shaders, but need to be specified in the following structure
		// *** Currently not being used
		VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
		pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
		pipelineLayoutInfo.setLayoutCount = 0;
		pipelineLayoutInfo.pSetLayouts = nullptr;
		pipelineLayoutInfo.pushConstantRangeCount = 0;
		pipelineLayoutInfo.pPushConstantRanges = nullptr;

		// Pipeline layout can now be created and stored in a class member
		if (vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &pipelineLayout) != VK_SUCCESS) {
			throw std::runtime_error("Failed to create pipeline layout!");
		}

		// The graphics pipeline create info can now be filled in, starting with the shader stages info
		VkGraphicsPipelineCreateInfo pipelineInfo{};
		pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
		pipelineInfo.stageCount = 2; // Number of shader stages
		pipelineInfo.pStages = shaderStages; // Pointer to the array of shader stages as defined earlier

		// Next we reference all the structures describing the fixed-function stage
		pipelineInfo.pVertexInputState = &vertexInputInfo;
		pipelineInfo.pInputAssemblyState = &inputAssembly;
		pipelineInfo.pViewportState = &viewportState;
		pipelineInfo.pRasterizationState = &rasterizer;
		pipelineInfo.pMultisampleState = &multisampling;
		pipelineInfo.pDepthStencilState = nullptr;
		pipelineInfo.pColorBlendState = &colourBlending;
		pipelineInfo.pDynamicState = &dynamicState;

		pipelineInfo.layout = pipelineLayout; // Pipeline layout reference

		pipelineInfo.renderPass = renderPass; // Render pass reference
		pipelineInfo.subpass = 0; // Index of the subpass this graphics pipeline will use

		// Graphics pipelines can derive from other pipelines if they share similar functionality
		pipelineInfo.basePipelineHandle = VK_NULL_HANDLE; // Parent graphics pipeline
		pipelineInfo.basePipelineIndex = -1; // Index of parent graphics pipeline about to be created

		// Finally, the graphics pipeline can be created
		// This creation function has a couple extra parameters to specify a pipeline cache and/or multiple pipelines
		if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &graphicsPipeline) != VK_SUCCESS) {
			throw std::runtime_error("Failed to create graphics pipeline!");
		}

		// Each shader module can now be deleted, since they are not required after the graphics pipeline has been created
		vkDestroyShaderModule(device, vertShaderModule, nullptr);
		vkDestroyShaderModule(device, fragShaderModule, nullptr);
	}

	void createFramebuffers() {
		/*
		* Creates a framebuffer for each image in the swap chain.
		* This is done by iterating through the previously created image views, and assigning each one to a newly created framebuffer.
		* Some other details are required as well from the swap chain images specification.
		*/
		swapChainFramebuffers.resize(swapChainImageViews.size()); // Resize to hold number of image views

		// Iterate through images views to create framebuffers from each of them
		for (size_t i = 0; i < swapChainImageViews.size(); i++) {
			// Image view to be used with the framebuffer
			VkImageView attachments[] = {
				swapChainImageViews[i]
			};

			// Fill in the framebuffer create info structure
			VkFramebufferCreateInfo framebufferInfo{};
			framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
			framebufferInfo.renderPass = renderPass; // Framebuffer only be used with render passes it is compatible with
			framebufferInfo.attachmentCount = 1;
			framebufferInfo.pAttachments = attachments; // Image views to be bound to the respective attachments in the render pass
			framebufferInfo.width = swapChainExtent.width;
			framebufferInfo.height = swapChainExtent.height;
			framebufferInfo.layers = 1; // Single layer images

			// Create the framebuffer in the specified image view index
			if (vkCreateFramebuffer(device, &framebufferInfo, nullptr, &swapChainFramebuffers[i]) != VK_SUCCESS) {
				throw std::runtime_error("Failed to create framebuffer!");
			}
		}
	}

	void createCommandPool() {
		/*
		* Creates the command pull, which manages memory used to store buffers and command buffers are allocated from them.
		* Command pool create infos only take two parameters, specifying how command buffers are rerecorded and what queue family to use.
		*/
		QueueFamilyIndices queueFamilyIndices = findQueueFamilies(physicalDevice);

		// Command pools only take two parameters
		VkCommandPoolCreateInfo poolInfo{};
		poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
		poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT; // Command buffers are rerecorded individually
		poolInfo.queueFamilyIndex = queueFamilyIndices.graphicsFamily.value(); // Can only allocate command buffers on a single type of queue
		// We have chosen the graphics queue family because we want to record commands for drawing

		// Now the command pool can be created
		if (vkCreateCommandPool(device, &poolInfo, nullptr, &commandPool) != VK_SUCCESS) {
			throw std::runtime_error("Failed to create command pool!");
		}
	}

	void createVertexBuffer() {
		/*
		* Creates a vertex buffer and writes our vertex data to it.
		* We first create a staging buffer to copy the data to, then transfer the data from that buffer over to the actual vertex buffer.
		* The vertex buffer is allocated from a memory type that is device local, which is most optimal for the GPU to read from.
		* But data cannot easily be copied over to buffers with this memory type, hence why we have to transfer it from the staging buffer.
		*/
		VkDeviceSize bufferSize = sizeof(vertices[0]) * vertices.size(); // Calculates the size needed for the buffer

		// Creates a staging buffer for mapping and copying the vertex data
		VkBuffer stagingBuffer;
		VkDeviceMemory stagingBufferMemory;
		createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory);

		// We can now write the vertex data to the staging buffer using memcpy
		void* data; // Pointer to the mapped memory
		vkMapMemory(device, stagingBufferMemory, 0, bufferSize, 0, &data); // Accesses a region of the specified memory resource defined by an offset and size
		memcpy(data, vertices.data(), (size_t)bufferSize); // Writes the vertex data to the mapped memory
		vkUnmapMemory(device, stagingBufferMemory); // Unmaps the memory since we no longer need to write to it

		// Creates the actual vertex buffer which is allocated from a memory type that is device local
		// This generally means vkMapMemory can not be used, however data from the staging buffer can be copied over to this buffer
		createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, vertexBuffer, vertexBufferMemory);

		copyBuffer(stagingBuffer, vertexBuffer, bufferSize); // Copies contents of staging buffer into vertex buffer

		// Staging buffer is no longer needed, so destroy buffer and free memory
		vkDestroyBuffer(device, stagingBuffer, nullptr);
		vkFreeMemory(device, stagingBufferMemory, nullptr);
	}

	void createIndexBuffer() {
		/*
		* Creates an index buffer and writes our index data to it.
		* Index buffers are used to reorder vertex data and reuse existing data for multiple vertices.
		* This is applicable when creating 3D meshes, as multiple triangles will be sharing vertices with eachother.
		* This function works almost identically to the above vertex buffer creation function.
		* (Refer to it for code explanation)
		*/
		VkDeviceSize bufferSize = sizeof(indices[0]) * indices.size();

		VkBuffer stagingBuffer;
		VkDeviceMemory stagingBufferMemory;
		createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory);

		void* data;
		vkMapMemory(device, stagingBufferMemory, 0, bufferSize, 0, &data);
		memcpy(data, indices.data(), (size_t)bufferSize);
		vkUnmapMemory(device, stagingBufferMemory);

		createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, indexBuffer, indexBufferMemory);

		copyBuffer(stagingBuffer, indexBuffer, bufferSize);

		vkDestroyBuffer(device, stagingBuffer, nullptr);
		vkFreeMemory(device, stagingBufferMemory, nullptr);
	}

	void createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties, VkBuffer& buffer, VkDeviceMemory& bufferMemory) {
		/*
		* Creates a buffer that holds data.
		* First we create a buffer that can store the data.
		* Then we obtain the buffer memory requirements and whether the GPU offers the type of memory the buffer needs.
		* Memory can then be allocated on the GPU and bound to the buffer.
		*/
		VkBufferCreateInfo bufferInfo{};
		bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
		bufferInfo.size = size; // Size of the buffer in bytes
		bufferInfo.usage = usage; // How the data in the buffer is going to be used
		bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE; // Like images in the swap chain, buffers can be owned by specific queue families

		// Vertex buffer can now be created
		if (vkCreateBuffer(device, &bufferInfo, nullptr, &buffer) != VK_SUCCESS) {
			throw std::runtime_error("Failed to create buffer!");
		}

		// We need to query the buffers memory requirements before we allocate memory to it
		VkMemoryRequirements memRequirements;
		vkGetBufferMemoryRequirements(device, buffer, &memRequirements);

		// Memory can now be allocated using this structure and providing the size and type of the memory
		VkMemoryAllocateInfo allocInfo{};
		allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
		allocInfo.allocationSize = memRequirements.size;
		allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties);

		// Allocate memory for the buffer
		if (vkAllocateMemory(device, &allocInfo, nullptr, &bufferMemory) != VK_SUCCESS) {
			throw std::runtime_error("Failed to allocate buffer memory!");
		}

		vkBindBufferMemory(device, buffer, bufferMemory, 0); // Binds the buffer memory to the buffer
		// The fourth parameter specifies the offset within the region of memory
	}

	void copyBuffer(VkBuffer srcBuffer, VkBuffer dstBuffer, VkDeviceSize size) {
		/*
		* Copies the contents of one buffer over to another.
		* Memory transfer operations are executed using command buffers much like drawing commands.
		* This implementation uses the existing command pool, but it may be more optimal to create a new temporary command pool.
		* Buffer copying requires a queue family that supports transfer operations, which luckily all families with the graphics or compute bit has.
		*/
		VkCommandBufferAllocateInfo allocInfo{};
		allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
		allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
		allocInfo.commandPool = commandPool; // Uses the existing command pool
		allocInfo.commandBufferCount = 1;

		// Create the command buffer to store transfer commands
		VkCommandBuffer commandBuffer;
		vkAllocateCommandBuffers(device, &allocInfo, &commandBuffer);

		VkCommandBufferBeginInfo beginInfo{};
		beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
		beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT; // Command buffer will only be used once

		vkBeginCommandBuffer(commandBuffer, &beginInfo); // Begin recording transfer commands

		// Record command to copy data from source buffer into destination buffer
		VkBufferCopy copyRegion{};
		copyRegion.srcOffset = 0; // Offset of source buffer region
		copyRegion.dstOffset = 0; // Offset of destination buffer region
		copyRegion.size = size; // Size of data to copy
		vkCmdCopyBuffer(commandBuffer, srcBuffer, dstBuffer, 1, &copyRegion);

		vkEndCommandBuffer(commandBuffer); // End after copy command has been recorded

		// Configure command buffer submit info
		VkSubmitInfo submitInfo{};
		submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
		submitInfo.commandBufferCount = 1;
		submitInfo.pCommandBuffers = &commandBuffer;

		vkQueueSubmit(graphicsQueue, 1, &submitInfo, VK_NULL_HANDLE); // Execute data transfer
		vkQueueWaitIdle(graphicsQueue); // Waits for the graphics queue to become idle (command execution finished)
		// Using a fence would allow us to schedule multiple transfers simultaneously and wait for all of them to complete (more optimal)

		vkFreeCommandBuffers(device, commandPool, 1, &commandBuffer); // Clean up temporary command buffer used for the transfer operation
	}

	uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties) {
		/*
		* Finds the type of memory to use for a vertex buffer.
		* This is done by using a bit field of memory types that are suitable in the typeFilter parameter.
		* We also need to check if the memory in the GPU can be written to from the CPU using the properties parameter.
		* These to values are checked against each memory type offered by the GPU to find a suitable index.
		*/
		// Queries the GPU for memory types it offers
		VkPhysicalDeviceMemoryProperties memProperties;
		vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProperties);

		// Checks each offered memory type from the GPU by the specifed type filter and whether we can write our vertex data to the memory
		for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
			if (typeFilter & (1 << i) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
				return i;
			}
		}

		throw std::runtime_error("Failed to find suitable memory type!"); // GPU offers no memory our vertex buffer can use
	}

	void createCommandBuffers() {
		/*
		* Allocates the command buffers.
		* Command buffers are allocated by specifying the command pool and number of buffers to allocate.
		* A command buffer is required for each frame in flight.
		*/
		commandBuffers.resize(MAX_FRAMES_IN_FLIGHT); // Resize to fit max frames in flight

		VkCommandBufferAllocateInfo allocInfo{};
		allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
		allocInfo.commandPool = commandPool;
		allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; // Can be submitted to a queue, but cannot be called from other command buffers
		allocInfo.commandBufferCount = (uint32_t)commandBuffers.size(); // Number of command buffers to allocate

		// Command buffers can now be allocated
		if (vkAllocateCommandBuffers(device, &allocInfo, commandBuffers.data()) != VK_SUCCESS) {
			throw std::runtime_error("Failed to allocate command buffers!");
		}
	}

	void recordCommandBuffer(VkCommandBuffer commandBuffer, uint32_t imageIndex) {
		/*
		* Records commands to a command buffer and the current swap chain image to write to.
		* Firstly, information about usage of the command buffer needs to be filled into a begin info structure.
		* Drawing then starts by beginning the render pass with some configuration options.
		* Commands being recorded are indicated by the vkCmd function prefix, and the first parameter must be the command buffer.
		* Also, all vkCmd functions return void, so no error handling can be performed during recording.
		* Graphics pipeline then has to be bound, then since viewport and scissor state were set to dynamic, they must be configured.
		* Vertex buffers holding data about the triangles are then bound to the bindings specifed in the graphics pipeline.
		* Finally, the draw command can be recorded, then the render pass and command buffer are finished.
		*/
		VkCommandBufferBeginInfo beginInfo{};
		beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
		beginInfo.flags = 0; // Can specify how the command buffer will be used
		beginInfo.pInheritanceInfo = nullptr; // Only relevant for secondary command buffers (specifies state to inherit from calling primary command buffer)

		// Command buffer recording can now begin
		if (vkBeginCommandBuffer(commandBuffer, &beginInfo) != VK_SUCCESS) {
			throw std::runtime_error("Failed to begin recording command buffer!");
		}

		// Begin render pass configuration options
		VkRenderPassBeginInfo renderPassInfo{};
		renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
		renderPassInfo.renderPass = renderPass; // Render pass which holds the attachments to bind
		renderPassInfo.framebuffer = swapChainFramebuffers[imageIndex]; // Framebuffer for swap chain image to draw to

		// Size of the render area needs to be defined, including the offset and extent
		renderPassInfo.renderArea.offset = { 0, 0 };
		renderPassInfo.renderArea.extent = swapChainExtent;

		// Defines the clear colour value for the attachment load operation clear
		VkClearValue clearColour = { { { 0.0f, 0.0f, 0.0f, 1.0f } } };
		renderPassInfo.clearValueCount = 1;
		renderPassInfo.pClearValues = &clearColour;

		// Render pass can now begin
		// Third parameter controls how the drawing commands within the render pass will be provided (primary or secondary)
		vkCmdBeginRenderPass(commandBuffer, &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);

		// Graphics pipeline can now be bound
		// Second parameter specifies if pipeline is graphics or compute
		vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipeline);

		// Set up viewport state
		VkViewport viewport{};
		viewport.x = 0.0f;
		viewport.y = 0.0f;
		viewport.width = static_cast<float>(swapChainExtent.width);
		viewport.height = static_cast<float>(swapChainExtent.height);
		viewport.minDepth = 0.0f;
		viewport.maxDepth = 1.0f;
		vkCmdSetViewport(commandBuffer, 0, 1, &viewport);

		// Set up scissor state
		VkRect2D scissor{};
		scissor.offset = { 0, 0 };
		scissor.extent = swapChainExtent;
		vkCmdSetScissor(commandBuffer, 0, 1, &scissor);

		// Bind vertex buffers to the binding specified in the graphics pipeline
		VkBuffer vertexBuffers[] = { vertexBuffer };
		VkDeviceSize offsets[] = { 0 };
		vkCmdBindVertexBuffers(commandBuffer, 0, 1, vertexBuffers, offsets);

		// Bind the index buffer (only one can be used)
		vkCmdBindIndexBuffer(commandBuffer, indexBuffer, 0, VK_INDEX_TYPE_UINT16);

		// Records the draw command for the square
		vkCmdDrawIndexed(commandBuffer, static_cast<uint32_t>(indices.size()), 1, 0, 0, 0);

		// Ends the render pass
		vkCmdEndRenderPass(commandBuffer);

		// After all commands are recorded, command buffer recording can end
		if (vkEndCommandBuffer(commandBuffer) != VK_SUCCESS) {
			throw std::runtime_error("Failed to record command buffer!");
		}
	}

	void createSyncObjects() {
		/*
		* Creates the semaphores and fences for each frame in flight.
		* They are created with the usual create info structures, although they don't take any fields in the current version of Vulkan.
		* At default, semaphores and fences are created in an unsignalled state.
		*/
		// Resize all semaphores and fences to fit max frames in flight
		imageAvailableSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
		renderFinishedSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
		inFlightFences.resize(MAX_FRAMES_IN_FLIGHT);

		VkSemaphoreCreateInfo semaphoreInfo{};
		semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

		VkFenceCreateInfo fenceInfo{};
		fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
		fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT; // Fence starts in signalled state

		// Creates the semaphores and fences for each frame in flight
		for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
			if (vkCreateSemaphore(device, &semaphoreInfo, nullptr, &imageAvailableSemaphores[i]) != VK_SUCCESS
				|| vkCreateSemaphore(device, &semaphoreInfo, nullptr, &renderFinishedSemaphores[i]) != VK_SUCCESS
				|| vkCreateFence(device, &fenceInfo, nullptr, &inFlightFences[i]) != VK_SUCCESS) {
				throw std::runtime_error("Failed to create semaphores and fences!");
			}
		}
	}

	void drawFrame() {
		/*
		* Draws a single frame.
		* It first waits for the last frame to finish using the previously created fence.
		* Then it obtains the next image in the swap chain and records the command buffer for it.
		* After configuration, the command buffers are submitted to the graphics queue for execution.
		* Once rendering of the image has finished, it can be queued up for presentation and shown to the screen.
		* If the window is resized or surface properties change, this function will pick that up and call a swap chain recreation.
		*/
		vkWaitForFences(device, 1, &inFlightFences[currentFrame], VK_TRUE, UINT64_MAX); // Waits for the previous frame to finish

		// Gets the next image in the swap chain and saves it's index to pick the VkFrameBuffer
		uint32_t imageIndex;
		VkResult result = vkAcquireNextImageKHR(device, swapChain, UINT64_MAX, imageAvailableSemaphores[currentFrame], VK_NULL_HANDLE, &imageIndex);

		// Checks if the swap chain is adequate for image processing (will not be if window size has changed)
		if (result == VK_ERROR_OUT_OF_DATE_KHR) {
			recreateSwapChain();
			return;
		} else if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR) {
			throw std::runtime_error("Failed to acquire swap chain image!");
		}

		vkResetFences(device, 1, &inFlightFences[currentFrame]); // Resets the fence to unsignalled state (only after we know swap chain is adequate)

		vkResetCommandBuffer(commandBuffers[currentFrame], 0); // Resets command buffer to ensure it is able to be recorded
		recordCommandBuffer(commandBuffers[currentFrame], imageIndex); // Records the commands for the specified image index

		// Queue submission and synchronisation is configured in this structure
		VkSubmitInfo submitInfo{};
		submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;

		// Specifies the semaphores to wait on
		VkSemaphore waitSemaphores[] = { imageAvailableSemaphores[currentFrame] }; // Waits for this semaphore before colour writing begins
		VkPipelineStageFlags waitStages[] = { VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT }; // Waits on the colour writing stage of the pipeline
		submitInfo.waitSemaphoreCount = 1;
		submitInfo.pWaitSemaphores = waitSemaphores; // Semaphores to wait on before execution
		submitInfo.pWaitDstStageMask = waitStages; // Stage(s) of the pipeline to wait (each entry corresponds to the semaphore in same index as above array)
		// Theoretically this implementation can already start executing the vertex shader while the image is not yet available

		// Specifies which command buffers to submit for execution
		submitInfo.commandBufferCount = 1;
		submitInfo.pCommandBuffers = &commandBuffers[currentFrame];

		// Specifies the semaphores to signal once the command buffer(s) have finished execution
		VkSemaphore signalSemaphores[] = { renderFinishedSemaphores[currentFrame] };
		submitInfo.signalSemaphoreCount = 1;
		submitInfo.pSignalSemaphores = signalSemaphores;

		// Submits the command buffer to the graphics queue, the specified fence is signalled once execution is complete
		if (vkQueueSubmit(graphicsQueue, 1, &submitInfo, inFlightFences[currentFrame]) != VK_SUCCESS) {
			throw std::runtime_error("Failed to submit draw command buffer!");
		}

		// Presentation of the frame is configured in this structure
		VkPresentInfoKHR presentInfo{};
		presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;

		// Specifies the semaphores to wait on before presentation
		presentInfo.waitSemaphoreCount = 1;
		presentInfo.pWaitSemaphores = signalSemaphores; // Waits for the image to finish rendering on the GPU

		// Specifies the swap chains to present images to and the index of the image for each swap chain
		VkSwapchainKHR swapChains[] = { swapChain };
		presentInfo.swapchainCount = 1;
		presentInfo.pSwapchains = swapChains;
		presentInfo.pImageIndices = &imageIndex; // Will almost always be a single image

		presentInfo.pResults = nullptr; // An array of VkResult can be specified here to check each swap chain if presentation was successful

		result = vkQueuePresentKHR(presentQueue, &presentInfo); // Submits the request to present an image to the swap chain

		// Checks if the swap chain is adequate for image presenting (will not be if window size or surface properties have changed)
		if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR || framebufferResized) {
			framebufferResized = false;
			recreateSwapChain();
		} else if (result != VK_SUCCESS) {
			throw std::runtime_error("Failed to present swap chain image!");
		}

		currentFrame = (currentFrame + 1) % MAX_FRAMES_IN_FLIGHT; // Advance to the next frame
	}

	VkShaderModule createShaderModule(const std::vector<char>& code) {
		/*
		* Creates a shader module for the graphics pipeline.
		* This requires a create info struct like any other Vulkan object.
		* Each shader module commonly represents bytecode for a single shader type, e.g. vertex, fragment, etc.
		*/
		VkShaderModuleCreateInfo createInfo{};
		createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
		createInfo.codeSize = code.size(); // Size of the code in bytes
		createInfo.pCode = reinterpret_cast<const uint32_t*>(code.data()); // Bytecode pointer needs to be converted to uint32_t rather than a char pointer

		// The shader module can now be created using the create info struct
		VkShaderModule shaderModule;
		if (vkCreateShaderModule(device, &createInfo, nullptr, &shaderModule) != VK_SUCCESS) {
			throw std::runtime_error("Failed to create shader module!");
		}

		return shaderModule;
	}

	VkSurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& availableFormats) {
		/*
		* Chooses the best surface format (colour depth) settings for the swap chain.
		* The best colour space to use is SRGB, as it results in more accurate perceived colours.
		* It's also the standard colour space for images.
		*/
		for (const VkSurfaceFormatKHR& availableFormat : availableFormats) { // Checks through each available format the device supports
			if (availableFormat.format == VK_FORMAT_B8G8R8A8_SRGB && availableFormat.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
				return availableFormat; // ^ Each colour channel is stored in an 8 bit unsigned integer, ^ Indicates if the SRGB colour space is supported
			}
		}

		// If the above check fails, it would be a better idea to rank the formats by how "good" they are
		return availableFormats[0]; // But for now, we can just choose the first format specified
	}

	VkPresentModeKHR chooseSwapPresentMode(const std::vector<VkPresentModeKHR>& availablePresentModes) {
		/*
		* Chooses the best present mode available for the swap chain.
		* The present mode is essentially how the "swapping" of images is going to work.
		* Basically, this function chooses whether to use vertical sync, triple buffering, or just immediate image presentation.
		* The best option would be to use triple buffering, since it avoids screen tearing while maintaining fairly low latency.
		* Although, all devices support standard vertical sync, so that is used instead if triple buffering is unavailable.
		*/
		for (const VkPresentModeKHR& availablePresentMode : availablePresentModes) { // Checks through each available present mode the device supports
			if (availablePresentMode == VK_PRESENT_MODE_MAILBOX_KHR) {
				return availablePresentMode; // ^ Triple buffering, best option for avoiding screen tearing while maintaining fairly low latency
			}
		}

		return VK_PRESENT_MODE_FIFO_KHR; // This is the only mode that is guaranteed to be available (mode is most similar to vertical sync)
	}

	VkExtent2D chooseSwapExtent(const VkSurfaceCapabilitiesKHR& capabilities) {
		/*
		* Chooses the swap extent for the swap chain.
		* This is the resolution of the swap chain images and is almost always equal to the resolution of the window.
		* If the current extent is not equal to the max uint32_t value, then it is fine to use.
		* Otherwise, the extent must be obtained from the GLFW function glfwGetFramebufferSize, which returns the window size in pixels.
		* Vulkan can only work with pixels, while GLFW does screen coordinates as well, which is why we use the above function to convert.
		* The values must then be clamped to the min/max extent values obtained from the device surface capabilites.
		*/
		if (capabilities.currentExtent.width != std::numeric_limits<uint32_t>::max()) {
			return capabilities.currentExtent; // Current extent is fine if the values do not equal the max uint32_t value
		} else { // Some window managers will set the extent to the max uint32_t value, in which case we pick the best resolution for the window within the min/max extent values
			int width, height;
			glfwGetFramebufferSize(window, &width, &height); // Gets the resolution of the window in pixels

			VkExtent2D actualExtent = {
				static_cast<uint32_t>(width),
				static_cast<uint32_t>(height)
			};

			// Both width and height values from the GLFW window pixel size must be clamped between the min/max values of the extent
			actualExtent.width = std::clamp(actualExtent.width, capabilities.minImageExtent.width, capabilities.maxImageExtent.width);
			actualExtent.height = std::clamp(actualExtent.height, capabilities.minImageExtent.height, capabilities.maxImageExtent.height);

			return actualExtent; // Resolution that best matches the window within the extent min/max values
		}
	}

	SwapChainSupportDetails queurySwapChainSupport(VkPhysicalDevice device) {
		/*
		* Used to populate the SwapChainSupportDetails struct.
		* This involves getting the basic surface capabilities, available formats, and supported presentation modes.
		*/
		SwapChainSupportDetails details;

		vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device, surface, &details.capabilities); // Gets the basic surface capabilities

		// Notice that when Vulkan returns a list of structs, we have to do this double function call approach
		uint32_t formatCount;
		vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, nullptr);

		if (formatCount != 0) {
			details.formats.resize(formatCount); // Vector needs to be resized to hold all available formats
			vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, details.formats.data()); // Gets the available formats
		}

		// Getting the supported present modes works exactly the same as above
		uint32_t presentModeCount;
		vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount, nullptr);

		if (presentModeCount != 0) {
			details.presentModes.resize(presentModeCount);
			vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount, details.presentModes.data()); // Gets supported present modes
		}

		return details;
	}

	bool isDeviceSuitable(VkPhysicalDevice device) {
		/*
		* Not all supported graphics cards are suitable for the operations we need to perform.
		* This function ensures that one is picked that is suitable.
		* A couple useful structs for obtaining useful device information are VkPhysicalDeviceProperties and VkPhysicalDeviceFeatures.
		* Using this information, there are many ways to choose a suitable graphics card.
		* A good way of choosing could be to give each device a score based on their properties and features, then choose the device with the highest score.
		* Some other properties that may be queuried are physical device extensions support and swap chain support. (especially useful for drawing images)
		*/
		QueueFamilyIndices indices = findQueueFamilies(device); // Check support for the queue families we require

		bool extensionsSupported = checkDeviceExtensionSupport(device); // Checks support for the physical device extensions we require

		bool swapChainAdequate = false;
		if (extensionsSupported) { // Only check for swap chain support if the extension is available
			SwapChainSupportDetails swapChainSupport = queurySwapChainSupport(device); // Gets swap chain support details for the physical device
			swapChainAdequate = !swapChainSupport.formats.empty() && !swapChainSupport.presentModes.empty(); // For now, swap chain support is sufficient if this is true
		}

		return indices.isComplete() && extensionsSupported && swapChainAdequate;
	}

	bool checkDeviceExtensionSupport(VkPhysicalDevice device) {
		/*
		* Checks the physical device for support for the device extensions we require.
		* This is done in an almost identical way to validation layers, so refer to the checkValidationLayerSupport function for more info.
		*/
		uint32_t extensionCount;
		vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, nullptr); // Get amount of extensions

		std::vector<VkExtensionProperties> availableExtensions(extensionCount);
		vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, availableExtensions.data()); // Get all available extensions

		std::set<std::string> requiredExtensions(deviceExtensions.begin(), deviceExtensions.end()); // Create a set of physical device extensions we require

		for (const VkExtensionProperties& extension : availableExtensions) {
			requiredExtensions.erase(extension.extensionName); // Removes the extension from the set if found
		}

		return requiredExtensions.empty(); // If set is empty, then all extensions are supported by the physical device
	}

	QueueFamilyIndices findQueueFamilies(VkPhysicalDevice device) {
		/*
		* Queue families hold a set of different queues that Vulkan commands can run on.
		* Each queue family allows only a subset of commands, so multiple queue families will need to be accessed and used.
		* We must check the physical device for the queue families that are supported by it, and then select the ones that suit our needs.
		*/
		QueueFamilyIndices indices; // A struct we created indicating the queue families we want to use

		// To no surprise, supported queue families are obtained the same way as extensions and validation layers
		uint32_t queueFamilyCount = 0;
		vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, nullptr);

		std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
		vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, queueFamilies.data());

		// We need to check each supported queue family to find the ones we require
		int i = 0; // The queue families we use are denoted by index, so we increment this every loop
		for (const VkQueueFamilyProperties& queueFamily : queueFamilies) {
			if (queueFamily.queueFlags & VK_QUEUE_GRAPHICS_BIT) { // Checks for a graphics queue family
				indices.graphicsFamily = i; // Assign the index to the graphics family
			}

			// Believe it or not, some graphics cards do not actually support surface rendering, so we have to check for that
			VkBool32 presentSupport = false;
			vkGetPhysicalDeviceSurfaceSupportKHR(device, i, surface, &presentSupport); // Checks whether the queue family at the specified index has surface support

			if (presentSupport) {
				indices.presentFamily = i; // Assign index if physical device has surface support
			}

			if (indices.isComplete()) {
				break; // An early exit if all requested queue families are found
			}

			i++;
		}

		return indices;
	}

	std::vector<const char*> getRequiredExtensions() {
		/*
		* Extensions are additional Vulkan functionality that need to be explicitly loaded in to be used.
		* For example, for Vulkan to actually work with GLFW, it needs to know the extensions that GLFW requires.
		* Other extensions can also be manually specified, as we do below with the debug messenger.
		* It's also important to note that functions within extensions must be manually loaded in, since they are not automatically.
		*/
		uint32_t glfwExtensionCount = 0; // Amount of extensions required
		const char** glfwExtensions; // An array of required extensions for GLFW
		glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount); // Returns an array of required extensions for GLFW for Vulkan to work
		// Also stores the amount of extensions in the provided parameter

		std::vector<const char*> extensions(glfwExtensions, glfwExtensions + glfwExtensionCount); // Converts the glfwExtensions array into a vector so more extensions can be added

		// Enables a debug messenger with a callback if validation layers are enabled
		if (enableValidationLayers) {
			extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME); // Adds the new extension to the vector
		}

		return extensions;
	}

	bool checkValidationLayerSupport() {
		/*
		* Validation layers are a feature of Vulkan that hook into Vulkan function calls to apply additional operations.
		* The most common operations are related to debugging, but can provide general info during runtime as well.
		* We can use vkEnumerateInstanceExtensionProperties to search for available validation layers.
		* This can then be used to check if the layers we have requested to use are available.
		*/
		uint32_t layerCount;
		vkEnumerateInstanceLayerProperties(&layerCount, nullptr); // Firstly, obtain the amount of available layers

		std::vector<VkLayerProperties> availableLayers(layerCount); // Creates a vector equal to the length of layerCount
		vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data()); // Call the function again to obtain all available layers by name

		// Now check for validation layer support
		for (const char* layerName : validationLayers) { // For each layer we have requested to use,
			bool layerFound = false;

			for (const VkLayerProperties& layerProperties : availableLayers) { // Check each available layer to see if it matches one
				if (strcmp(layerName, layerProperties.layerName) == 0) { // This line uses a C function to compare the two strings
					layerFound = true;
					break;
				}
			}

			if (!layerFound) {
				return false; // If a layer is not found, return false
			}
		}

		return true; // If all layers are available, return true
	}

	static std::vector<char> readFile(const std::string& filename) {
		/*
		* A helper function that reads a file given the file name.
		*/
		// Opens the file given two flags: ate starts reading it from the end, and binary reads it as a binary file.
		std::ifstream file(filename, std::ios::ate | std::ios::binary);

		if (!file.is_open()) { // Checks if the file has opened properly
			throw std::runtime_error("Failed to open file!");
		}

		// Advantage of reading from the end is we can determine the size of the file from this position
		size_t fileSize = (size_t)file.tellg(); // Gets position of current character
		std::vector<char> buffer(fileSize); // Allocate buffer with obtained size

		file.seekg(0); // Returns back to the beginning character of the file
		file.read(buffer.data(), fileSize); // Puts the file contents into the buffer

		file.close(); // File can now be closed

		return buffer; // Returns contents of file
	}

	static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity, VkDebugUtilsMessageTypeFlagsEXT messageType, const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData, void* pUserData) {
		/*
		* This is a Vulkan debug callback function with the following parameters:
		* messageSeverity: The severity of the message, where we can filter out messages we don't care about.
		* messageType: The type of the message, ranging from specification/performance, possible mistake, or non-optimal use of Vulkan.
		* pCallbackData: Refers to a struct that contains many details about the message itself.
		* pUserData: A pointer specified during the setup of the callback, allowing us to provide our own data to it.
		* This callback returns a boolean indicating if the Vulkan call that triggered the validation layer message should be aborted.
		*/
		std::cerr << "Validation Layer: " << pCallbackData->pMessage << std::endl;

		return VK_FALSE;
	}
};

int main() {
	HelloTriangleApplication app;

	try {
		app.run();
	} catch (const std::exception& e) {
		std::cerr << e.what() << std::endl;
		return EXIT_FAILURE;
	}

	return EXIT_SUCCESS;
}