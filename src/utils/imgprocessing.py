import pandas as pd
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2
from tabulate import tabulate
from skimage.measure import shannon_entropy
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
from tqdm import tqdm

        


class ImageProcessor():
    # class provides methods to prepare images for modelling

    @staticmethod
    def from_np_to_pil(img_np: np.ndarray,
                    assume_float01: bool = True,
                    make_rgb: bool = False) -> Image.Image:
        """
        Convert a NumPy image (H,W[,C]) → PIL.Image.Image.

        * Floats are scaled to 0-255.
        * (H,W,1) is squeezed to (H,W)   →  "L" mode
        or broadcast to (H,W,3)        →  "RGB" mode if make_rgb=True.
        """
        arr = np.asarray(img_np)

        # ----- handle dtype ------------------------------------------------------
        if arr.dtype != np.uint8:
            if assume_float01:                       # floats already in [0,1]
                arr = np.clip(arr, 0.0, 1.0) * 255.0
            else:                                    # min-max stretch
                imin, imax = arr.min(), arr.max()
                arr = 0 if imin == imax else (arr - imin) / (imax - imin) * 255.0
            arr = arr.astype(np.uint8)

        # ----- fix shape ---------------------------------------------------------
        if arr.ndim == 3 and arr.shape[2] == 1:      # (H,W,1) ➜ grayscale or RGB
            if make_rgb:
                arr = np.repeat(arr, 3, axis=2)      # (H,W,3)
            else:
                arr = arr.squeeze(axis=2)            # (H,W)

        return Image.fromarray(arr)
    
    
    
    
    
    @staticmethod
    def list_files(from_dir, extensions=('.png', ".jpeg", ".jpg", )):
        """
        lists all files in a directory and returns a list with file names
        """        
        if not os.path.exists(from_dir):
            raise Exception(f"directory {from_dir} does not exist")
        
        # get all files in the directory
        filenames = os.listdir(from_dir)  # Returns a list of filenames
        # filter by extensions
        filenames = [f for f in filenames if f.endswith(extensions)]

        return filenames
    
    
       

    @staticmethod
    def load_images(from_dir, imgNames="all", extensions=('.png', ".jpeg", ".jpg", )):
        """
        loads images by file names like 'Viral Pneumonia-101.png' from a directory

        params: imgNames is a list with file names
        returns: a tuple with loaded images and a tuple with corresponding image names
        """
        imgs = []
                
        if not imgNames:
            raise Exception("provide a list with image names including the extension")
        
        # to load all images from a directory
        if imgNames == "all":
            imgNames = ImageProcessor.list_files(from_dir, extensions=extensions)
            
        for iname in imgNames: 
            
            # determine image path
            fileDir = from_dir
            filePath = os.path.join(fileDir, iname)
            if os.path.exists(filePath):
                img = cv2.imread(filePath, cv2.IMREAD_GRAYSCALE)
                imgs.append(img)
            else:
                raise Exception(f"file {iname} not found in {fileDir}")
                
        return tuple(imgs), tuple(imgNames)       



    @staticmethod
    def store(img, name, to_dir):
        """
        stores image to a directory with the given name
        """
        if not os.path.exists(to_dir):
            os.makedirs(to_dir)
        
        filePath = os.path.join(to_dir, name)
        cv2.imwrite(filePath, img)

        return filePath
    
    
     
    
    
    @staticmethod
    def downscale_to_dir(inputFolder, outputFolder, new_size, interpolation=cv2.INTER_AREA):
        """
        downscales all images in a directory to the new size and stores them in the output folder
        """
        # load images
        iNames = ImageProcessor.list_files(inputFolder)
        imgs, iNames = ImageProcessor.load_images(imgNames=iNames, from_dir=inputFolder)
        for img_name, img in zip(iNames, imgs):
            
            # Downscale the image
            if new_size is not None:
                img = cv2.resize(img, new_size, interpolation=interpolation)
            
            # store the image
            ImageProcessor.store(img, img_name, outputFolder)
            




    @staticmethod
    def rotate_to_dir(inputFolder, outputFolder, rotAngle):
        
        if rotAngle not in [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_180, cv2.ROTATE_90_COUNTERCLOCKWISE]:
            raise Exception(f"rotation angle {rotAngle} not supported")
        
        # load images
        iNames = ImageProcessor.list_files(inputFolder)
        imgs, iNames = ImageProcessor.load_images(imgNames=iNames, from_dir=inputFolder)
        
        angleStr = {cv2.ROTATE_90_CLOCKWISE: "90 degrees clockwise",
                     cv2.ROTATE_180: "180 degrees",
                     cv2.ROTATE_90_COUNTERCLOCKWISE: "90 degrees counterclockwise"}
        
        print(f"rotating {len(imgs)} images by {angleStr[rotAngle]}\nfrom {inputFolder} \nto {outputFolder}")
        
        for img_name, img in zip(iNames, imgs):
            
            # Rotate the image
            img_rot = cv2.rotate(img, rotAngle)
                
            # store the image
            ImageProcessor.store(img_rot, img_name, outputFolder)
  
  


    @staticmethod
    def copyRenameDownscaleRotateSave( imgs, imgNames, to_dir, new_resolution, strip_prefix, add_prefix, rotAngle):
        
        os.makedirs(to_dir, exist_ok=True)
        cnt = 0
        rotated_names = []

        for img, name in zip(imgs, imgNames):
            
            img_rot = None
            
            # Downscale the image
            if new_resolution is not None:
                img = cv2.resize(img, new_resolution, interpolation=cv2.INTER_AREA)

            # Rotate the image
            if rotAngle != 0:
                img_rot = cv2.rotate(img, rotAngle)
            else:
                img_rot = img

            # Strip prefix if present
            if name.startswith(strip_prefix):
                stripped_name = name[len(strip_prefix):]
            else:
                stripped_name = name

            # Add new prefix
            new_name = add_prefix + stripped_name
            rotated_names.append(new_name)
            # Define full source and destination paths
            # src_path = os.path.join(from_dir, name)
            dst_path = os.path.join(to_dir, new_name)
            # save the rotated image
            cv2.imwrite(dst_path, img_rot)
            # Copy the file
            # shutil.copy2(src_path, dst_path)
            cnt += 1
        
        print(f"{cnt} images have been copied and rotated to {to_dir}")
        
        return rotated_names



    @staticmethod
    def plot_images(images, titles=None, tSize=10, max_img_per_row=5):
        
        import math 
        
        num_images = len(images)
        #cols = min(5, num_images)  # Maximum 5 images per row
        cols = min(max_img_per_row, num_images)  # Maximum 4 images per row
        rows = math.ceil(num_images / cols)  # Calculate required rows

        fig_width = cols * 4  # Set width dynamically (4 inches per image)
        fig_height = rows * 4  # Set height dynamically (4 inches per row)

        fig, axes = plt.subplots(rows, cols, figsize=(fig_width, fig_height))

        # Flatten axes array if there's more than one row
        if rows > 1:
            axes = axes.flatten()
        else:
            axes = [axes] if num_images == 1 else axes

        for i, img in enumerate(images):
            axes[i].imshow(img, cmap='gray')  # Display image (adjust cmap as needed)
            if titles:
                axes[i].set_title(titles[i], fontsize=tSize)
            axes[i].axis("off")  # Hide axes

        # Hide unused subplots
        for j in range(i + 1, len(axes)):
            axes[j].axis("off")

        plt.tight_layout()
        # doesn't block the execution in main thread
        plt.show(block=False)
        
    
    
    @staticmethod
    def downscale(img, new_size, interpolation=cv2.INTER_AREA, plotResult=True):
        # resizes a gray-scaled image to the new_size and returns
              
        # Resize using INTER_AREA
        downscaled_image = cv2.resize(img, new_size, interpolation=interpolation)

        if plotResult:
            try :
                # Display the original and overlay images
                plt.title(f"Downscaled image {new_size}")
                plt.imshow(downscaled_image)
                plt.axis('off')
                plt.show()  
            except:
                pass
        
        return downscaled_image



    @staticmethod
    def downscaleToFolder(inputFolder, outputFolder, new_size, interpolation=cv2.INTER_AREA, debug=True):
        # overwrites already existing files if the name is equal
        # get image names
        filenames = ImageProcessor.list_files(inputFolder)  # Returns a list of filenames
        
        # load images
        imgs, iNames = ImageProcessor.load_images(imgNames=filenames, from_dir=inputFolder)
        
        # create directory
        os.makedirs(outputFolder, exist_ok=True)
        
        cnt = 0

        for img, name in zip(imgs, iNames):
            if img is None:
                print(f"[Warning] Skipping image (could not load): {name}")
                continue

            try:
                dImg = ImageProcessor.downscale(img, new_size, interpolation, plotResult=False)

                # replace extension with .png
                base_name = os.path.splitext(name)[0]
                new_filename = base_name + ".png"
                file_path = os.path.join(outputFolder, new_filename)

                cv2.imwrite(file_path, dImg)
                print(f"[Saved] {file_path}")
                cnt += 1
            except Exception as e:
                print(f"[Error] Could not process {name}: {e}")

        
        if debug:
            print(f"{cnt} images have been downscaled and stored to {outputFolder}")       
        
        return cnt  
            
                  
  
    @staticmethod
    def getRoiWithResizedMask( img, mask, contourThiknes=1, plotResult=True):
        # Resize the mask to match the X-ray image size and returns the original image 
        # overlayed with mask contour 
        
        # Resize the mask to match the X-ray image size
        resized_mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)

        # Find contours from the resized mask
        contours, _ = cv2.findContours(resized_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Create a copy of the X-ray image for overlay
        overlay_image = img.copy()

        # Draw contours on the overlay image in red (BGR: (0, 0, 255))
        cv2.drawContours(overlay_image, contours, -1, (0, 0, 255), thickness=contourThiknes)

        # Convert images to RGB for matplotlib visualization
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        overlay_image_rgb = cv2.cvtColor(overlay_image, cv2.COLOR_BGR2RGB)
        
        
        # Create a binary mask for the contour area
        binary_mask = cv2.drawContours(
            np.zeros_like(resized_mask, dtype=np.uint8),  # Blank canvas
            contours,
            -1,
            (255),  # White color for contours
            thickness=contourThiknes
        )

        # Fill the contours to include the area inside
        filled_mask = cv2.drawContours(
            binary_mask,
            contours,
            -1,
            (255),  # White color for filling
            thickness=cv2.FILLED
        )

        # Apply the mask to the original image
        lung_area = cv2.bitwise_and(img, img, mask=filled_mask)


        if plotResult:
            try :
                # Display the original and overlay images
                plt.figure(figsize=(15, 5))
                plt.subplot(1, 4, 1)
                plt.title("Original X-ray (299x299)")
                plt.imshow(img_rgb)
                plt.axis('off')
                
                plt.subplot(1, 4, 2)
                plt.title("Resized mask (299x299)")
                plt.imshow(resized_mask)
                plt.axis('off')

                plt.subplot(1, 4, 3)
                plt.title("X-ray with Mask's contours")
                plt.imshow(overlay_image_rgb)
                plt.axis('off')
                
                plt.subplot(1, 4, 4)
                plt.title("Region of interest (ROI) inc. contour")
                plt.imshow(lung_area)
                plt.axis('off')

                plt.tight_layout()
                plt.show()  
                
            except:
                pass
        
        return lung_area



    # Analyze class distribution
    @staticmethod
    def class_distribution(image_directories):
        class_counts = {category: len(os.listdir(paths["images"])) for category, paths in image_directories.items()}
        plt.bar(class_counts.keys(), class_counts.values(), color='green')
        plt.title('Class Distribution')
        #plt.xlabel('Category')
        plt.ylabel('Number of Images')
        plt.show()
    

    @staticmethod
    def get_image_metadata(image_directory):
        # Dictionary to store image properties
        image_metadata = {}

        # Iterate through all .png images in the directory
        all_images = [img for img in os.listdir(
            image_directory) if img.endswith('.png')]

        for image_name in all_images:
            image_path = os.path.join(image_directory, image_name)

            # Read the image
            image = cv2.imread(image_path)
            if image is None:
                print(f"Warning: Could not read image {image_name}")
                continue

            # Get resolution (height, width, channels)
            height, width = image.shape[:2]

            # Determine color property
            if len(image.shape) == 3 and image.shape[2] == 3:
                color_property = "RGB"
            else:
                color_property = "grey"

            # Add to dictionary
            image_metadata[image_name] = [f"{width}x{height}", color_property]

        return image_metadata


    @staticmethod
    def img_data_overview(image_directories):
        data = []
        # "Check for completeness of images, masks, and masked images."
        # Initialize counters for Images, Masks, and Masked
        total_counts = [0, 0, 0]

        for key, value in image_directories.items():
            cnts = []
            for k, folder_path in value.items():
                count = len(os.listdir(folder_path))
                cnts.append(count)

            # Add counts for the current category to the total
            total_counts = [total_counts[i] + cnts[i] for i in range(len(cnts))]

            # Append category-specific data
            data.append([key] + cnts)

        # Append totals to the data
        data.append(["Total"] + total_counts)

        # Headers
        headers = ["Category", "Images", "Masks", "Masked"]

        # Print side by side
        print(tabulate(data, headers=headers, tablefmt="grid"))


    @staticmethod
    def img_count(categories, image_directories):
        # Use specified categories or all by default
        cntDict = {}
        for category in categories:
            image_directory = image_directories[category]["images"]
            # Count the number of images in each directory
            cntDict[category] = len([img for img in os.listdir(
                image_directory) if img.endswith('.png')])

        return cntDict


    @staticmethod
    def has_black_frame(image, threshold=10):
        """Check if the image has a black frame around it."""
        height, width = image.shape[:2]

        # Extract borders
        top_border = image[:threshold, :]
        bottom_border = image[-threshold:, :]
        left_border = image[:, :threshold]
        right_border = image[:, -threshold:]

        # Combine all borders
        borders = np.concatenate((top_border.flatten(), bottom_border.flatten(
        ), left_border.flatten(), right_border.flatten()))

        # If most border pixels are black, return True
        return np.mean(borders) < 10


    @staticmethod
    def calculate_blurriness(image):
        # Variance of Laplacian method for blurriness detection
        return cv2.Laplacian(image, cv2.CV_64F).var()


    @staticmethod
    def calculate_contrast(image):
        # Contrast calculated as the difference between max and min pixel intensities
        return image.max() - image.min()


    @staticmethod
    def calculate_variance(image):
        # Variance of pixel intensities
        return np.var(image)


    @staticmethod
    def calculate_entropy(image):
        # Entropy calculation using skimage.measure.shannon_entropy
        return shannon_entropy(image)


    @staticmethod
    def get_greyscale_image_metrics(directory):
        # List to store image properties
        image_metrics = []
        this = ImageProcessor

        # Iterate through all images in the directory
        all_images = [img for img in os.listdir(
            directory) if img.endswith('.png') or img.endswith('.jpg')]

        for image_name in all_images:
            image_path = os.path.join(directory, image_name)

            # Load the image in greyscale
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                print(f"Warning: Could not read image {image_name}")
                continue

            # Calculate image metrics
            mean_intensity = np.mean(image)
            variance = this.calculate_variance(image)
            blurriness = this.calculate_blurriness(image)
            contrast = this.calculate_contrast(image)
            entropy = this.calculate_entropy(image)

            # Append properties to the list
            image_metrics.append({
                "file name": image_name.strip(".png"),
                "mean intensity": mean_intensity,
                "variance": variance,
                "blurriness": blurriness,
                "contrast": contrast,
                "entropy": entropy
            })

        # Convert list to DataFrame
        df = pd.DataFrame(image_metrics)

        return df


    @staticmethod
    def rename_masks(masks_dir):
        # Iterate through all files in the directory
        for filename in os.listdir(masks_dir):
            if filename.startswith("m"):  # already renamed
                continue
            else:
                # Construct the old and new file paths
                old_path = os.path.join(masks_dir, filename)
                new_filename = f"m{filename}"
                new_path = os.path.join(masks_dir, new_filename)

                # Rename the file
                os.rename(old_path, new_path)
                print(f"Renamed: {filename} -> {new_filename}")



    @staticmethod
    def generate_masked_images(from_dir, model, ori_confs, ori_cols, select_imgs="all", target_size=(256, 256)):
        """
        Generate masked images for chest X-ray images using a pre-trained model.

        Parameters:
        - from_dir (str): Path to the folder containing input images.
        - model (tf.keras.Model): Pre-trained segmentation model.
        - target_size (tuple): Image size (height, width) for resizing.

        Returns:
        - masked_images (tuple): Tuple of (masked_image_array, name)
        - names (tuple): Tuple of original image names
        """
        
        imgs, names = ImageProcessor.load_images(from_dir=from_dir, imgNames=select_imgs)
        images = []
        for i, col in enumerate(np.round(ori_confs, 2)):
            match ori_cols[np.argmax(col)]:
                case "rotated_180":
                    image = cv2.rotate (imgs[i], cv2.ROTATE_180)
                case "rotated_90":
                    image = cv2.rotate(imgs[i], cv2.ROTATE_90_COUNTERCLOCKWISE)
                case "rotated_minus_90":
                    image = cv2.rotate(imgs[i], cv2.ROTATE_90_CLOCKWISE)
                case "rotated_0":
                    image = imgs[i]
            images.append(image)

        masked_images = []

        for img in tqdm(images, total=len(imgs)):
            
            # resize masked image to target size
            img = cv2.resize(img, target_size, interpolation=cv2.INTER_LINEAR)
            img_array = img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)  # (1, h, w, 1)
            prediction = model.predict(img_array)
            mask = (prediction[0, :, :, 0] > 0.5).astype(np.uint8) * 255  # Convert to 0-255
            mask = (mask > 127).astype(np.uint8)
            
            masked = np.asarray(img).copy() * mask
            masked_images.append(masked)

        return masked_images, names



    @staticmethod
    def visualize_random_images(image_paths, n_samples_per_category=5):
        """
        Visualize random samples of images from each category.

        Args:
            image_paths (list of tuples): List of tuples with (category, path) pairs.
            n_samples_per_category (int): Number of random samples to visualize per category.
        """
        # Group images by category
        category_dict = {}
        for category, path in image_paths:
            if category not in category_dict:
                category_dict[category] = []
            category_dict[category].append(path)

        # Visualize
        n_categories = len(category_dict)
        fig, axes = plt.subplots(
            n_categories, n_samples_per_category, figsize=(20, 4 * n_categories))

        if n_categories == 1:
            axes = [axes]  # Ensure axes is iterable when there is only one category

        for row, (category, paths) in enumerate(category_dict.items()):
            sampled_paths = np.random.choice(
                paths, size=n_samples_per_category, replace=False)
            for col, path in enumerate(sampled_paths):
                img = Image.open(path).convert('L')  # Convert to grayscale
                axes[row][col].imshow(img, cmap='gray')
                axes[row][col].set_title(category)
                axes[row][col].axis('off')

        plt.tight_layout()
        plt.show()
