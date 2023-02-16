#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <iostream>
#include <vector>
#include <optional>
#include <set>
#include <algorithm>

// Window sizes
const uint32_t WIDTH = 800;
const uint32_t HEIGHT = 600;

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

        createSurface(); // Creates the surface for Vulkan to present rendered images

        pickPhysicalDevice(); // Chooses a graphics card to use for Vulkan

        createLogicalDevice(); // Creates the logical device for Vulkan

        createSwapChain(); // Creates the swap chain for Vulkan

        createImageViews(); // Creates the image views for each image in the swap chain
    }

    void mainLoop() {
        // Runs as long as the window is open
        while (!glfwWindowShouldClose(window)) {
            glfwPollEvents(); // Checks for input events
        }
    }

    void cleanup() {
        for (VkImageView imageView : swapChainImageViews) {
            vkDestroyImageView(device, imageView, nullptr); // Unlike images, image views were explicitly created by us, hence they need to be destroyed
        }

        vkDestroySwapchainKHR(device, swapChain, nullptr); // Destroys the swap chain
        vkDestroyDevice(device, nullptr); // Destroys the logical device

        if (enableValidationLayers) {
            DestroyDebugUtilsMessengerEXT(instance, debugMessenger, nullptr); // Destroys the Vulkan debug messenger
        }

        vkDestroySurfaceKHR(instance, surface, nullptr); // Destroys the Vulkan surface
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
        }
        else {
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
        }
        else {
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
        }
        else { // Some window managers will set the extent to the max uint32_t value, in which case we pick the best resolution for the window within the min/max extent values
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