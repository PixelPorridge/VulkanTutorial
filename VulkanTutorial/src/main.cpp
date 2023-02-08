#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <iostream>
#include <vector>
#include <optional>

// Window sizes
const uint32_t WIDTH = 800;
const uint32_t HEIGHT = 600;

// Validation layers allow the program to be error checked during runtime
// Here we can create a vector and specify the Vulkan SDK built-in layers that we want
const std::vector<const char*> validationLayers = {
    "VK_LAYER_KHRONOS_validation" // All the useful standard validation
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
    }
    else {
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
    std::optional<uint32_t> graphicsFamily;

    bool isComplete() {
        // Checks that all required properties in the struct have a value, aka the queue families are supported by the graphics card
        return graphicsFamily.has_value();
    }
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

    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE; // The graphics card Vulkan will be using
    VkDevice device; // The logical device Vulkan will be using

    VkQueue graphicsQueue; // The graphics queue handle

    void initWindow() {
        glfwInit(); // This must be called to initialise the GLFW library

        // glfwWindowHint sets options for the window
        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API); // Tells GLFW to not create an OpenGL context, since that's what it was first built to do
        glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE); // Disables the ability to resize the window, since it's a little difficult to handle in Vulkan

        window = glfwCreateWindow(WIDTH, HEIGHT, "Vulkan", nullptr, nullptr); // Creates a GLFW window, simple enough
    }

    void initVulkan() {
        createInstance(); // Creates an instance for Vulkan

        setupDebugMessenger(); // Sets up a debug messenger for Vulkan

        pickPhysicalDevice(); // Chooses a graphics card to use for Vulkan

        createLogicalDevice(); // Creates the logical device for Vulkan
    }

    void mainLoop() {
        // Runs as long as the window is open
        while (!glfwWindowShouldClose(window)) {
            glfwPollEvents(); // Checks for input events
        }
    }

    void cleanup() {
        vkDestroyDevice(device, nullptr); // Destroys the logical device

        if (enableValidationLayers) {
            DestroyDebugUtilsMessengerEXT(instance, debugMessenger, nullptr); // Destroys the Vulkan debug messenger
        }

        vkDestroyInstance(instance, nullptr); // Destroys the Vulkan instance

        glfwDestroyWindow(window); // Destroys the window and its context

        glfwTerminate(); // Terminates the GLFW library
    }

    void createInstance() {
        // Before any Vulkan code is run, the requested validation layers should be checked for availability
        if (enableValidationLayers && !checkValidationLayerSupport()) {
            throw std::runtime_error("Validation layers requested, but not available!");
        }

        /*
        * In order to create an instance, some structs must be filled out first.
        * The compulsory struct is VkInstanceCreateInfo, and VkApplicationInfo is optional, but recommended.
        * Once VkInstanceCreateInfo is created, it is then provided to vkCreateInstance to create an instance for Vulkan.
        * This is generally how objects in Vulkan are created.
        */
        VkApplicationInfo appInfo{}; // A struct to hold information about the application
                                     // While optional, it essentially provides the driver with useful information to optimise the application
        appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO; // Many structs for Vulkan require you to explicitly specify the type
        appInfo.pApplicationName = "Hello Triangle";
        appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
        appInfo.pEngineName = "No Engine";
        appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
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
        }
        else {
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
        createInfo.pfnUserCallback = debugCallback; // More info about the avobe parameters can be found in this function
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
            throw std::runtime_error("failed to set up debug messenger!");
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

        // Loops through the supported devices to find one that is suitable for our uses
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
        * The first task is to specify the queues Vulkan will be using, in this case, only the graphics queue.
        * This is also where device features can be enabled for use.
        * Lastly, much like the Vulkan instance object, it requires a main create info struct to be filled in.
        * Devices can also use extensions and validation layers, but unlike the Vulkan instance, these are device specific.
        * Validation layers for the device are ignored by up-to-date implementations, since the Vulkan instance handles these anyways.
        * Although, it is still recommended to specify these to be compatible with older implementations.
        */
        QueueFamilyIndices indices = findQueueFamilies(physicalDevice); // Accesses the queue families for the physical device

        // Multiple of these queue create info structs can be specified, depending on how many queue families we are using
        VkDeviceQueueCreateInfo queueCreateInfo{}; // Describes the number of queues we want for a single queue family
        queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        queueCreateInfo.queueFamilyIndex = indices.graphicsFamily.value(); // We're only interested in a queue with graphics capabilities for now
        queueCreateInfo.queueCount = 1; // So we also set the queue count to one

        float queuePriority = 1.0f;
        queueCreateInfo.pQueuePriorities = &queuePriority; // Influences the scheduling of command buffer execution (required even for one queue)

        VkPhysicalDeviceFeatures deviceFeatures{}; // Specifies the set of device features we want to use (leaving this blank for now)

        // Specifies the create info for the logical device, it has similar fields to the Vulkan instance one
        VkDeviceCreateInfo createInfo{}; // The main device create info struct
        createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;

        createInfo.pQueueCreateInfos = &queueCreateInfo; // Points to the queue create info structs
        createInfo.queueCreateInfoCount = 1; // Amount of queue create infos (when multiple queue families are used)

        createInfo.pEnabledFeatures = &deviceFeatures; // Points to the enabled device features

        createInfo.enabledExtensionCount = 0; // We aren't using any device specific extensions

        // (Optional) These fields are ignored in up-to-date implementations of Vulkan,
        // but it's still recommended to specify these anyways to be compatible with older implementations
        if (enableValidationLayers) {
            createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
            createInfo.ppEnabledLayerNames = validationLayers.data();
        }
        else {
            createInfo.enabledLayerCount = 0;
        }

        // The logical device can now be created
        if (vkCreateDevice(physicalDevice, &createInfo, nullptr, &device) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create logical device!");
        }

        // The queues are automatically created with the logical device, but this creates handles to interface with them
        vkGetDeviceQueue(device, indices.graphicsFamily.value(), 0, &graphicsQueue); // Creating one queue from the graphics queue family
    }

    bool isDeviceSuitable(VkPhysicalDevice device) {
        /*
        * Not all supported graphics cards are suitable for the operations we need to perform.
        * This function ensures that one is picked that is suitable.
        * A couple useful structs for obtaining useful device information are VkPhysicalDeviceProperties and VkPhysicalDeviceFeatures.
        * Using this information, there are many ways to choose a suitable graphics card.
        * A good way of choosing could be to give each device a score based on their properties and features, then choose the device with the highest score.
        */
        VkPhysicalDeviceProperties deviceProperties; // (Optional) Holds basic device properties like the name, type and supported Vulkan version
        vkGetPhysicalDeviceProperties(device, &deviceProperties);

        VkPhysicalDeviceFeatures deviceFeatures; // (Optional) Holds optional features like texture compression, 64 bit floats and multi viewport rendering (useful for VR)
        vkGetPhysicalDeviceFeatures(device, &deviceFeatures);

        // Since we're just starting out, all we'll check for is support for the queue families we want to use
        QueueFamilyIndices indices = findQueueFamilies(device);

        return indices.isComplete(); // Checks if all requested queue families are supported by the graphics card
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
        * (Optional) It's possible to check for all available extensions.
        * This can be done by first finding the amount of available extensions, then creating a vector to hold these extensions by name.
        * Using the vkEnumerateInstanceExtensionProperties, this is possible.
        * These can then be output to the console.
        */
        uint32_t availableExtensionCount = 0;
        vkEnumerateInstanceExtensionProperties(nullptr, &availableExtensionCount, nullptr); // First, obtain the amount of available extensions

        std::vector<VkExtensionProperties> availableExtensions(availableExtensionCount); // Creates a vector of length equal to extensionCount
        vkEnumerateInstanceExtensionProperties(nullptr, &availableExtensionCount, availableExtensions.data()); // Call the function again to obtain the extensions by name

        // Output the available extensions to the console
        std::cout << "Available Extensions:\n";
        for (const VkExtensionProperties& availableExtension : availableExtensions) {
            std::cout << '\t' << availableExtension.extensionName << '\n';
        }

        /*
        * Now here we can get the required extensions for GLFW and any additional extensions we want.
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
        * Previously we used vkEnumerateInstanceExtensionProperties to search for available extensions.
        * In an almost identical way to extensions, we can search for available validation layers.
        * This can then be used to check if the layers we have requested to use are available.
        */
        uint32_t layerCount;
        vkEnumerateInstanceLayerProperties(&layerCount, nullptr); // Firstly, obtain the amount of available layers

        std::vector<VkLayerProperties> availableLayers(layerCount); // Creates a vector equal to the length of layerCount
        vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data()); // Call the function again to obtain all available layers by name

        // (Optional) Before we check for support, we can output all the available validation layers to the console
        std::cout << "Available Validation Layers:\n";
        for (const VkLayerProperties& layer : availableLayers) {
            std::cout << '\t' << layer.layerName << '\n';
        }

        // Now check for validation layer support
        for (const char* layerName : validationLayers) { // For each layer we have requested to use,
            bool layerFound = false;

            for (const auto& layerProperties : availableLayers) { // Check each available layer to see if it matches one
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
    }
    catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}