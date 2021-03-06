#include <iostream>
#include <time.h>
#include <stdlib.h>
#include <math.h>

#define DEBUG_ZZX 0

//GLEW
#define GLEW_STATIC
#include <glew.h>

//GLFW
#include <glfw3.h>

// GLM Mathematics
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

// Other includes
#include "Shader.h"
#include "Camera.h"

//Window dimensions
const GLuint WIDTH = 1000, HEIGHT = 1000;

// Camera
Camera camera(glm::vec3(0.0f, 0.0f, 5.0f));
bool keys[1024];

GLfloat deltaTime = 0.0f;
GLfloat lastFrame = 0.0f;

static int state = 0;//used to control what is shown in the screen
static char rotate_state = '0';//used to control rotation

//used to represent a patch
struct model
{
	float * para;               //parameters
	int       num_inliers;  //total number of inliers
	int    * inliers_index; //which points are inliers of this patch
	float * inliers_res;     //residue of each inliers to this patch
	float    scale;              //scale of noise of all inliers
	float    score;              //used to mearsure how good this patch is
	bool     bSig;               //used to indicate whether this patch is a significant one 
	int       group;            //which cluster this patch belongs
};

//used when all points are sorted based on their corresponding residue to a specific patch
struct point_res
{
	int    point_index;
	float res;
};

struct vertex_cluster
{
	int    num_neighbor;
	int * neighbor;
	int    num_rep;
	float* rep_patch_para;
	int    cluster_index;
	bool bOutlier; //true if it is considered as an outlier
};

//Function prototypes
//Callback
void gl_Key_Callback(GLFWwindow* window, int key, int scancode, int action, int mode);
void scroll_callback(GLFWwindow* window, double xoffset, double yoffset);
void Do_Movement();
//others
void calc_patch_score(struct model * patch);
void clustering_over_perm(struct model * patch_init, float * res_init, int * sig_set, int num_sig, int num_points);
void inst_models(float * inst, float * vertices, float * surface_paras, int iDim, int num_clusters, int ** cluster_vertex_table, int * cluster_vertex_table_index);
void least_squares(float * vertices, float * pfModel, int * cons_set, int cons_set_size, int iDim);
void merge_cluster(struct vertex_cluster * vertex_array, int * cluster_size, int iDim, float * cluster, int iNumPoints, int num_clusters, int * cluster_table, float thres_coeff, float thres_dist);
void mergesort(float * array, int len);
void mergesort_with_index(struct point_res * array, int len);
void search_neighbor(struct vertex_cluster * vertex_array, int index_cur, int index_parent, int cluster, int iDim, float thres_coeff, float thres_dist);
float calc_dist(float * pfCurVert, float * pfModel, int iDim);
float ikose(float * res, int &num_inliers, int k, float E);
int    find_neighbor(int index_cur, float * vertices, int iDim, int iNumPoints, int * neighbor_index, float range);
int    gen_init_patches(float * vertices, int iDim, int iNumPoints, float * models, int iNumModelsMax);
int    otd(float * scores, int len);
int    subspace_modeling(float * vertices, int iDim, int iNumPoints, float * patch_para, int iNumModelsMax);


int main()
{
	srand(time(NULL));//random seed initialization

	printf("-------------------------------------------------------------------------------------------------\n");
	printf(" Shape Detection from Raw LiDAR Data with Subspace Modeling\n");
	printf("Zhenxun Zhuang,  First year PhD @ CS, Stony Brook University\n");
	printf("-------------------------------------------------------------------------------------------------\n");

	long start_time, end_time, elapsed_min, elapsed_sec;

	start_time = clock();

	//GLFW Initialization
	glfwInit();
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);//openGL 3.3
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
	glfwWindowHint(GLFW_RESIZABLE, GL_FALSE);

	//Create a GLFW window and set it to be the current context
	GLFWwindow * window = glfwCreateWindow(WIDTH, HEIGHT, "FirstShow", nullptr, nullptr);
	if (window == nullptr)
	{
		std::cout << "Failed to create a GLFW window!" << std::endl;
		glfwTerminate();
		return -1;
	}
	glfwMakeContextCurrent(window);

	glfwSetKeyCallback(window, gl_Key_Callback);
	glfwSetScrollCallback(window, scroll_callback);

	//GLEW Initialization
	glewExperimental = GLFW_TRUE;
	if (glewInit() != GLEW_OK)
	{
		std::cout << "Failed to initialize GLEW!" << std::endl;
		return -1;
	}

	//Define size of the window shown on screen
	int iWidth = 0, iHeight = 0;
	glfwGetFramebufferSize(window, &iWidth, &iHeight);
	glViewport(0, 0, iWidth, iHeight);

	//--------------------------------------------------------------read vertex data------------------------------------------------------------------
	end_time = clock();
	elapsed_min = (end_time - start_time) / (CLOCKS_PER_SEC * 60);
	elapsed_sec = (end_time - elapsed_min * 60 * CLOCKS_PER_SEC - start_time) / CLOCKS_PER_SEC;

	printf("%02ld:%02ld Please input the point cloud data file path: ", elapsed_min, elapsed_sec);
	char inputDataPath[100]; //"..\\..\\..\\Data\\data.csv";
	gets_s(inputDataPath, 99);

	FILE * fInputData = NULL;
	errno_t err;
	err = fopen_s(&fInputData, inputDataPath, "r");
	if (err)
	{
		printf("Cannot access input data file!\n");
	}

	int iDim = 0, iNumPoints = 0;
	fscanf_s(fInputData, "%d,%d", &iDim, &iNumPoints);
	if (iDim == 3)
	{
		int temp = 0;
		fscanf_s(fInputData, ",%d", &temp);
	}
	int iVertSize = iDim*iNumPoints;
	GLfloat * vertices = new GLfloat[iVertSize];
	GLfloat * vertices_ind = vertices;
	GLfloat maxabs = 0.0;
	for (int i = 0; i < iNumPoints; i ++)
	{
		for (int j = 0; j < iDim-1; j++)
		{
			fscanf_s(fInputData, "%f,", vertices_ind);//suppose input is a csv file
			if (fabsf(*vertices_ind) > maxabs)
			{
				maxabs = fabsf(*vertices_ind);
			}
			vertices_ind++;
		}
		fscanf_s(fInputData, "%f", vertices_ind);
		if (fabsf(*vertices_ind) > maxabs)
		{
			maxabs = fabsf(*vertices_ind);
		}
		vertices_ind++;
	}
	fclose(fInputData);

	for (int i = 0; i < iVertSize; i++)
	{
		vertices[i] = vertices[i] / maxabs;//force input data to be in [-1,1]
	}

	end_time = clock();
	elapsed_min = (end_time - start_time) / (CLOCKS_PER_SEC * 60);
	elapsed_sec = (end_time - elapsed_min * 60 * CLOCKS_PER_SEC - start_time) / CLOCKS_PER_SEC;
	printf("%02ld:%02ld Successfully read the whole point cloud which contains %d points!\n", elapsed_min, elapsed_sec, iNumPoints);


	Shader MyShader_2D("VertexShader_2D.txt", "FragmentShader.txt");
	Shader MyShader_3D("VertexShader_3D.txt", "FragmentShader.txt");
	

	//Generate vertex data buffer
	GLuint VBO_pointCloud;
	glGenBuffers(1, &VBO_pointCloud);
	//Generate VAO
	GLuint VAO_pointCloud;
	glGenVertexArrays(1, &VAO_pointCloud);
	//Bind vertex array object
	glBindVertexArray(VAO_pointCloud);
	//Bind vertex data buffer
	glBindBuffer(GL_ARRAY_BUFFER, VBO_pointCloud);
	//Copy vertex data
	glBufferData(GL_ARRAY_BUFFER, sizeof(GLfloat)*iVertSize, vertices, GL_STATIC_DRAW);
	//Link vertex data with vertex attribute
	glVertexAttribPointer(0, iDim, GL_FLOAT, GL_FALSE, iDim * sizeof(GLfloat), (const void *)0);
	glEnableVertexAttribArray(0);
	//Unbind vertex data buffer
	//Note that this is allowed, the call to glVertexAttribPointer registered VBO as the currently bound vertex buffer object so afterwards we can safely unbind
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	//Unbind vertex array object
	glBindVertexArray(0);


	//---------------------------------------------find representative patches for each point--------------------------------------------
	struct vertex_cluster * vertex_array = new struct vertex_cluster[iNumPoints];

	int * neighbor_index = new int[iNumPoints];//which vertices are in the neighbor
	float * vertices_cur = new float[iNumPoints*iDim];//neighboring vertices

	int iNumModelsMax = 40;//Maximum initial patch numbers
	float * rep_patch_para_temp = new float[iNumModelsMax*(iDim + 1)];//parameters of current point

	//make sure the neighbor of each vertex contains around 100 points
	//float num_points_per_region = 50;
	//float num_regions = iNumPoints / num_points_per_region;
	//float num_regions_per_dim = powf(num_regions, 1.0 / iDim);
	//float range = 1.0 / num_regions_per_dim;
	//float range = 0.08;

	end_time = clock();
	elapsed_min = (end_time - start_time) / (CLOCKS_PER_SEC * 60);
	elapsed_sec = (end_time - elapsed_min * 60 * CLOCKS_PER_SEC - start_time) / CLOCKS_PER_SEC;
	printf("%02ld:%02ld Finding representative patches for point ", elapsed_min, elapsed_sec);

	for (int i = 0; i < iNumPoints; i++)
	{
		if (i > 0)
		{
			for (int backn = 0; backn < 6; backn++)
			{
				printf("\b");
			}
		}
		printf("%06d", i);

		int num_neighbor = 0;
		for (float range = 0.002; range < 0.03; range += 0.001)
		{
			num_neighbor = find_neighbor(i, vertices, iDim, iNumPoints, neighbor_index, range);
			if (num_neighbor >= 100)
			{
				break;
			}
		}

		if (num_neighbor <= 3)//too far away from clusters, discard
		{
			vertex_array[i].bOutlier = true;
			vertex_array[i].num_neighbor = 0;
			continue;
		}

		vertex_array[i].num_neighbor = num_neighbor;
		vertex_array[i].neighbor = new int[num_neighbor];
		for (int j = 0; j < num_neighbor; j++)
		{
			vertex_array[i].neighbor[j] = neighbor_index[j];
		}

		for (int j = 0; j < num_neighbor; j++)
		{
			for (int k = 0; k < iDim; k++)
			{
				vertices_cur[j*iDim + k] = vertices[neighbor_index[j] * iDim + k];
			}
		}

		int num_rep = subspace_modeling(vertices_cur, iDim, num_neighbor, rep_patch_para_temp, iNumModelsMax);

		if (num_rep == 0)//no representative patch means outliers
		{
			vertex_array[i].bOutlier = true;
			vertex_array[i].num_rep = 0;
			continue;
		}

		vertex_array[i].num_rep = num_rep;
		vertex_array[i].rep_patch_para = new float[(iDim + 1)*num_rep];
		for (int j = 0; j < (iDim + 1)*num_rep; j++)
		{
			vertex_array[i].rep_patch_para[j] = rep_patch_para_temp[j];
		}

		vertex_array[i].bOutlier = false;
	}

	end_time = clock();
	elapsed_min = (end_time - start_time) / (CLOCKS_PER_SEC * 60);
	elapsed_sec = (end_time - elapsed_min * 60 * CLOCKS_PER_SEC - start_time) / CLOCKS_PER_SEC;
	printf("\n%02ld:%02ld Successfully find representative patches for each point!\n", elapsed_min, elapsed_sec);

	//--------------------------------------------------clustering-----------------------------------------------------------------------
	for (int i = 0; i < iNumPoints; i++)
	{//initialization
		if (vertex_array[i].bOutlier)
		{
			vertex_array[i].cluster_index = i;//outliers won't go through clustering process
		}
		else
		{
			vertex_array[i].cluster_index = -1;
		}
	}

	int * cluster_size = new int[iNumPoints];//size of each cluster
	int num_clusters = 0;
	int * cluster_table = new int[iNumPoints];//move all clusters together
	int * cluster_index_switch_table = new int[iNumPoints];//position of each cluster in cluster_table
	int ** cluster_vertex_table = NULL;
	int * cluster_vertex_table_index = NULL;
	float * surfaces_para = NULL;

	int ite_num = 20;
	int cluster_sig_thres = 0;
	float thres_coeff_st = 0.2, thres_dist_st = 0.1;
	float thres_coeff_int = (thres_coeff_st - 0.10) / ite_num;
	float thres_dist_int   = (thres_dist_st - 0.05) / ite_num;

	for (int ite = 0; ite <= ite_num; ite++)
	{
		float thres_coeff = thres_coeff_st - thres_coeff_int *ite;
		float thres_dist = thres_dist_st - thres_dist_int  *ite;

		if (ite == 0)
		{
			for (int i = 0; i < iNumPoints; i++)
			{
				if (vertex_array[i].cluster_index == -1)
				{
					search_neighbor(vertex_array, i, i, i, iDim, thres_coeff, thres_dist);
				}
			}

			for (int i = 0; i < iNumPoints; i++)
			{
				cluster_size[i] = 0;
			}
			for (int i = 0; i < iNumPoints; i++)
			{
				cluster_size[vertex_array[i].cluster_index]++;
			}

			cluster_sig_thres = 20;
		}
		else
		{
			merge_cluster(vertex_array, cluster_size, iDim, surfaces_para, iNumPoints, num_clusters, cluster_table, thres_coeff, thres_dist);

			cluster_sig_thres = ite+20;
		}

		//clean last time
		if (num_clusters > 0)
		{
			for (int i = 0; i < num_clusters; i++)
			{
				delete[] cluster_vertex_table[i];
			}
			delete[] cluster_vertex_table;
			delete[] cluster_vertex_table_index;
			delete[] surfaces_para;
		}

		//new round
		num_clusters = 0;
		for (int i = 0; i < iNumPoints; i++)
		{
			cluster_index_switch_table[i] = -1;
		}
		for (int i = 0; i < iNumPoints; i++)
		{
			if (cluster_size[i] >= cluster_sig_thres)//used to filter out weak clusters
			{
				cluster_index_switch_table[i] = num_clusters;
				cluster_table[num_clusters] = i;
				num_clusters++;
			}
		}

		cluster_vertex_table = new int*[num_clusters];//vertices belonging to each cluster
		cluster_vertex_table_index = new int[num_clusters];//current pointer position of each row in cluster_vertex_table
		for (int i = 0; i < num_clusters; i++)
		{
			cluster_vertex_table[i] = new int[cluster_size[cluster_table[i]]];
			cluster_vertex_table_index[i] = 0;
		}
		for (int i = 0; i < iNumPoints; i++)
		{
			int cluster_index_cur = vertex_array[i].cluster_index;
			int pos = cluster_index_switch_table[cluster_index_cur];
			if (pos != -1)
			{
				cluster_vertex_table[pos][cluster_vertex_table_index[pos]] = i;
				cluster_vertex_table_index[pos]++;
			}
		}

		surfaces_para = new float[num_clusters*(iDim + 1)];//parameters of each model
		for (int i = 0; i < num_clusters; i++)
		{
			least_squares(vertices, surfaces_para + i*(iDim + 1), cluster_vertex_table[i], cluster_size[cluster_table[i]], iDim);
		}
	}

	int num_vertices_per_model = (iDim == 2) ? 2 : 6;
	float * inst = new float[num_vertices_per_model*iDim*num_clusters];
	inst_models(inst, vertices, surfaces_para, iDim, num_clusters, cluster_vertex_table, cluster_vertex_table_index);

	end_time = clock();
	elapsed_min = (end_time - start_time) / (CLOCKS_PER_SEC * 60);
	elapsed_sec = (end_time - elapsed_min * 60 * CLOCKS_PER_SEC - start_time) / CLOCKS_PER_SEC;
	printf("%02ld:%02ld Successfully reconstruct the whole scene, %d planes are detected!\n", elapsed_min, elapsed_sec, num_clusters);

	for (int i = 0; i < num_clusters; i++)
	{
		float * para_cur = surfaces_para + (iDim + 1) * i;
		printf("%d:   ", i);
		if (iDim == 2)
		{
			if (para_cur[1] < 0.5)
			{//x=c
				printf("x = %.4f\n", para_cur[2]);
			}
			else
			{//y=kx+b
				printf("y = %.4f*x + %.4f\n", -para_cur[0], para_cur[2]);
			}
		}
		else if (iDim == 3)
		{
			if (para_cur[2] < 0.5)
			{
				if (para_cur[1] < 0.5)
				{//x=d
					printf("x = %.4f\n", para_cur[3]);
				}
				else
				{//ax+y=d
					printf("%.4fx + y = %.4f\n", para_cur[0], para_cur[3]);
				}
			}
			else
			{//ax+by+cz=d
				printf("%.4fx + %.4fy + z = %.4f\n", para_cur[0], para_cur[1], para_cur[3]);
			}
		}
	}

	//Generate vertex data buffer
	GLuint VBO_inst_patches;
	glGenBuffers(1, &VBO_inst_patches);
	//Generate VAO
	GLuint VAO_inst_patches;
	glGenVertexArrays(1, &VAO_inst_patches);
	//Bind vertex array object
	glBindVertexArray(VAO_inst_patches);
	//Bind vertex data buffer
	glBindBuffer(GL_ARRAY_BUFFER, VBO_inst_patches);
	//Copy vertex data
	glBufferData(GL_ARRAY_BUFFER, sizeof(GLfloat) * num_vertices_per_model * iDim * num_clusters, inst, GL_STATIC_DRAW);
	//Link vertex data with vertex attribute
	glVertexAttribPointer(0, iDim, GL_FLOAT, GL_FALSE, iDim * sizeof(GLfloat), (const void *)0);
	glEnableVertexAttribArray(0);
	//Unbind vertex data buffer
	//Note that this is allowed, the call to glVertexAttribPointer registered VBO as the currently bound vertex buffer object so afterwards we can safely unbind
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	//Unbind vertex array object
	glBindVertexArray(0);

	//clean
	delete[] vertices;
	delete[] vertices_cur;
	delete[] rep_patch_para_temp;

	for (int i = 0; i < iNumPoints; i++)
	{
		if (!vertex_array[i].bOutlier)
		{
			delete[] vertex_array[i].neighbor;
			delete[] vertex_array[i].rep_patch_para;
		}
	}
	delete[] vertex_array;
	delete[] neighbor_index;
#if 1
	delete[] cluster_size;
	delete[] cluster_table;
	delete[] cluster_index_switch_table;
	for (int i = 0; i < num_clusters; i++)
	{
		delete[] cluster_vertex_table[i];
	}
	if (num_clusters > 0)
	{
		delete[] cluster_vertex_table;
		delete[] cluster_vertex_table_index;

		delete[] surfaces_para;

		delete[] inst;
	}

#endif

	//--------------------------------------------------show time-----------------------------------------------------------------------
	printf("-------------------------------------------------------------------------------------------------\n");
	printf("Press following buttons to show results of each step:\n");
	printf("0 --- Point Cloud\n");
	printf("1 --- Reconstruction\n");
	printf("-------------------------------------------------------------------------------------------------\n");
	if (iDim == 3)
	{
		printf("Press following buttons to change the view\n");
		printf("Arrow ---------------- Move along respective direction\n");
		printf("F -------------------- Move towards you\n");
		printf("B -------------------- Move away from you\n");
		printf("X, Y, Z -------------- Clockwisely rotate around respective axis\n");
		printf("Shift + X, Y, Z ------ Counter-clockwisely rotate around respective axis\n");
		printf("O -------------------- Undo all rotations\n");
		printf("Mouse Wheel ---------- Zoom in/out\n");
		printf("-------------------------------------------------------------------------------------------------\n");
	}

	glPointSize(1.0);
	glLineWidth(4.0);
	
	GLint vertexColorLocation = 0;
	if (iDim == 2)
	{
		vertexColorLocation = glGetUniformLocation(MyShader_2D.Program, "myColor");
	}
	else if (iDim == 3)
	{
		vertexColorLocation = glGetUniformLocation(MyShader_3D.Program, "myColor");
	}

	glEnable(GL_DEPTH_TEST);

	glm::mat4 trans;
	GLfloat trans_degree = 2.0f;
	GLint transLoc;
	// Create camera transformation
	glm::mat4 model;
	glm::mat4 view;
	glm::mat4 projection;
	GLint modelLoc;
	GLint viewLoc;
	GLint projLoc;

	//Loop
	while (!glfwWindowShouldClose(window))
	{
		// Set frame time
		GLfloat currentFrame = glfwGetTime();
		deltaTime = currentFrame - lastFrame;
		lastFrame = currentFrame;

		// Check if any events have been activated (key pressed, mouse moved etc.) and call corresponding response functions
		glfwPollEvents();
		Do_Movement();

		glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		//Finally...
		if (iDim == 2)
		{
			MyShader_2D.Use();

			if (state >= 0)
			{
				glUniform4f(vertexColorLocation, 0.5f, 0.5f, 0.5f, 1.0f);
				glBindVertexArray(VAO_pointCloud);
				glDrawArrays(GL_POINTS, 0, iNumPoints);
			}
			if (state == 1)
			{
				glUniform4f(vertexColorLocation, 1.0f, 0.0f, 0.0f, 1.0f);
				glBindVertexArray(VAO_inst_patches);
				glDrawArrays(GL_LINES, 0, num_vertices_per_model * num_clusters);
			}
		}
		else if (iDim == 3)
		{
			MyShader_3D.Use();

			view = camera.GetViewMatrix();
			projection = glm::perspective(glm::radians(camera.Zoom), (float)WIDTH / (float)HEIGHT, 1.0f, 20.0f);
			// Get the uniform locations
			modelLoc = glGetUniformLocation(MyShader_3D.Program, "model");
			viewLoc = glGetUniformLocation(MyShader_3D.Program, "view");
			projLoc = glGetUniformLocation(MyShader_3D.Program, "projection");
			// Pass the matrices to the shader
			glUniformMatrix4fv(modelLoc, 1, GL_FALSE, glm::value_ptr(model));
			glUniformMatrix4fv(viewLoc, 1, GL_FALSE, glm::value_ptr(view));
			glUniformMatrix4fv(projLoc, 1, GL_FALSE, glm::value_ptr(projection));

			if (state >= 0)
			{
				glUniform4f(vertexColorLocation, 1.0f, 1.0f, 1.0f, 1.0f);
				glBindVertexArray(VAO_pointCloud);
				glDrawArrays(GL_POINTS, 0, iNumPoints);
			}
			if (state == 1)
			{
				glBindVertexArray(VAO_inst_patches);
				for (int i = 0; i < num_clusters; i++)
				{
					glUniform4f(vertexColorLocation, i / (float)num_clusters, cosf(float(i)), 1.0f - 0.5*i / (float)num_clusters, 0.0f);
					glDrawArrays(GL_TRIANGLES, num_vertices_per_model*i, num_vertices_per_model);
				}
			}


			if (rotate_state == 'x')
			{
				trans = glm::rotate(trans, glm::radians(trans_degree), glm::vec3(1.0, 0.0, 0.0));
			}
			else if (rotate_state == 'y')
			{
				trans = glm::rotate(trans, glm::radians(trans_degree), glm::vec3(0.0, 1.0, 0.0));
			}
			else if (rotate_state == 'z')
			{
				trans = glm::rotate(trans, glm::radians(trans_degree), glm::vec3(0.0, 0.0, 1.0));
			}
			else if (rotate_state == 'X')
			{
				trans = glm::rotate(trans, glm::radians(-trans_degree), glm::vec3(1.0, 0.0, 0.0));
			}
			else if (rotate_state == 'Y')
			{
				trans = glm::rotate(trans, glm::radians(-trans_degree), glm::vec3(0.0, 1.0, 0.0));
			}
			else if (rotate_state == 'Z')
			{
				trans = glm::rotate(trans, glm::radians(-trans_degree), glm::vec3(0.0, 0.0, 1.0));
			}
			else if (rotate_state == 'o')
			{
				trans = glm::mat4();
			}
			transLoc = glGetUniformLocation(MyShader_3D.Program, "transform");
			glUniformMatrix4fv(transLoc, 1, GL_FALSE, glm::value_ptr(trans));

			rotate_state = '0';
		}

		glBindVertexArray(0);

		glfwSwapBuffers(window);
	}

	//Cleaning
	glDeleteVertexArrays(1, &VAO_pointCloud);
	glDeleteBuffers(1, &VBO_pointCloud);
	glDeleteVertexArrays(1, &VAO_inst_patches);
	glDeleteBuffers(1, &VBO_inst_patches);

	//The end
	glfwTerminate();

	printf("Thank you for using this platform!\n");
	printf("-------------------------------------------------------------------------------------------------\n");

	return 0;
}

void gl_Key_Callback(GLFWwindow* window, int key, int scancode, int action, int mode)
{
	if (key == GLFW_KEY_ESCAPE && action == GLFW_RELEASE)
	{
		glfwSetWindowShouldClose(window, GL_TRUE);
	}

	if (key == GLFW_KEY_0 && action == GLFW_PRESS)
	{
		state = 0;//point clouds
	}
	else if (key == GLFW_KEY_1 && action == GLFW_PRESS)
	{
		state = 1;//intial patches
	}
	else if (key == GLFW_KEY_2 && action == GLFW_PRESS)
	{
		state = 2;//significant patches
	}
	else if (key == GLFW_KEY_3 && action == GLFW_PRESS)
	{
		state = 3;//clusters
	}
	else if (key == GLFW_KEY_4 && action == GLFW_PRESS)
	{
		state = 4;//representative patches
	}
	else if (key == GLFW_KEY_5 && action == GLFW_PRESS)
	{
		state = 5;//To be determined sometime later
	}

	if (key >= 0 && key < 1024)
	{
		if (action == GLFW_PRESS)
			keys[key] = true;
		else if (action == GLFW_RELEASE)
			keys[key] = false;
	}

	if (keys[GLFW_KEY_X])
	{
		if (mode == GLFW_MOD_SHIFT)
		{
			rotate_state = 'X';
		}
		else
		{
			rotate_state = 'x';
		}
	}
	else if (keys[GLFW_KEY_Y])
	{
		if (mode == GLFW_MOD_SHIFT)
		{
			rotate_state = 'Y';
		}
		else
		{
			rotate_state = 'y';
		}
	}
	else if (keys[GLFW_KEY_Z])
	{
		if (mode == GLFW_MOD_SHIFT)
		{
			rotate_state = 'Z';
		}
		else
		{
			rotate_state = 'z';
		}
	}
	else if (key == GLFW_KEY_O && action == GLFW_RELEASE)
	{
		rotate_state = 'o';
	}
}

// Moves/alters the camera positions based on user input
void Do_Movement()
{
	// Camera controls
	if (keys[GLFW_KEY_UP])
		camera.ProcessKeyboard(UP, deltaTime);
	if (keys[GLFW_KEY_DOWN])
		camera.ProcessKeyboard(DOWN, deltaTime);
	if (keys[GLFW_KEY_LEFT])
		camera.ProcessKeyboard(LEFT, deltaTime);
	if (keys[GLFW_KEY_RIGHT])
		camera.ProcessKeyboard(RIGHT, deltaTime);
	if (keys[GLFW_KEY_F])
		camera.ProcessKeyboard(FORWARD, deltaTime);
	if (keys[GLFW_KEY_B])
		camera.ProcessKeyboard(BACKWARD, deltaTime);
}

void scroll_callback(GLFWwindow* window, double xoffset, double yoffset)
{
	camera.ProcessMouseScroll(yoffset);
}

void model_rectify(float * para, int num_model, int iDim)
{
	float thres = 5e-2, times = 20.0;

	if (iDim == 2)
	{//ax+by=c
		for (int i = 0; i < num_model; i++)
		{
			float * para_cur = para + i*(iDim + 1);

			float * a = para_cur;
			float * b = para_cur + 1;
			float * c = para_cur + 2;

			if (fabs(*a) > times)//x=c
			{
				*c = *c / (*a);
				*a = 1.0;
				*b = 0.0;
			}
			else if (fabs(*a) < thres)
			{
				*a = 0.0;
			}
		}
	}//line
	else if (iDim == 3)
	{//ax+by+cz=d
		for (int i = 0; i < num_model; i++)
		{
			float * para_cur = para + i*(iDim + 1);

			float * a = para_cur;
			float * b = para_cur + 1;
			float * c = para_cur + 2;
			float * d = para_cur + 3;

			if (fabs(*a) < thres)
			{
				*a = 0.0;
			}
			else if (fabs(*b) < thres)
			{
				*b = 0.0;
			}

			if (fabs(*c) < thres)
			{//ax+y=d
				if (fabs(*a) > times)
				{//x=d
					*d = *d / (*a);
					*a = 1.0;
					*b = 0.0;
					*c = 0.0;
				}
			}
			else
			{
				if ((fabs(*a) > times) || (fabs(*b) > times))
				{
					*c = 0.0;

					if ((fabs(*a) < thres) || ((fabs(*a) > thres) && (fabs(*b / (*a)) > times)))
					{//y=d
						*a = 0.0;
						*d = *d / (*b);
						*b = 1.0;
					}
					else if ((fabs(*b) < thres) || ((fabs(*b) > thres) && (fabs(*a / (*b)) > times)))
					{//x=d
						*b = 0.0;
						*d = *d / (*a);
						*a = 1.0;
					}
					else//ax+y=d
					{
						*d = *d / (*b);
						*a = *a / (*b);
						*b = 1.0;
					}
				}
			}//c==1.0
		}//end loop for each model
	}//plane
}

void inst_models(float * inst, float * vertices, float * surface_paras, int iDim, int num_clusters, int ** cluster_vertex_table, int * cluster_vertex_table_index)
{
	int num_vertices_per_model = (iDim == 2) ? 2 : 6;

	for (int clu = 0; clu < num_clusters; clu++)
	{
		int num_neighbor = cluster_vertex_table_index[clu];

		float xmin = 2, xmax = -2;
		float ymin = 2, ymax = -2;
		float zmin = 2, zmax = -2;
		for (int i = 0; i < num_neighbor; i++)
		{
			float x = vertices[iDim*cluster_vertex_table[clu][i]];
			if (x < xmin)
			{
				xmin = x;
			}
			if(x>xmax)
			{
				xmax = x;
			}

			float y = vertices[iDim*cluster_vertex_table[clu][i] + 1];
			if (y < ymin)
			{
				ymin = y;
			}
			if (y > ymax)
			{
				ymax = y;
			}

			if (iDim == 3)
			{
				float z = vertices[iDim*cluster_vertex_table[clu][i] + 2];
				if (z < zmin)
				{
					zmin = z;
				}
				if (z > zmax)
				{
					zmax = z;
				}
			}
		}


		if (iDim == 2)//lines
		{
			float * pfModelTemp = surface_paras + clu*(iDim + 1);
			float a = *pfModelTemp;
			float b = *(pfModelTemp + 1);
			float c = *(pfModelTemp + 2);

			float * pfInstTemp = inst + clu * num_vertices_per_model * iDim;
			if (fabs(b) < 1e-5)//x=c
			{
				*pfInstTemp = c;
				*(pfInstTemp + 1) = ymin;
				*(pfInstTemp + 2) = c;
				*(pfInstTemp + 3) = ymax;
			}
			else
			{//ax+by=c
				*pfInstTemp = xmin;
				*(pfInstTemp + 1) = -a*xmin + c;
				*(pfInstTemp + 2) = xmax;
				*(pfInstTemp + 3) = -a*xmax + c;
			}
		}
		else if (iDim == 3)//planes
		{
			float * pfModelTemp = surface_paras + clu*(iDim + 1);
			float a = *pfModelTemp;
			float b = *(pfModelTemp + 1);
			float c = *(pfModelTemp + 2);
			float d = *(pfModelTemp + 3);

			float * pfVertTemp = inst + clu * num_vertices_per_model * iDim;
			if (fabs(c) < 1e-5)
			{
				if (fabs(b) < 1e-5)
				{//x=d
				 //first triangle
					pfVertTemp[0] = d;
					pfVertTemp[1] = ymin;
					pfVertTemp[2] = zmin;

					pfVertTemp[3] = d;
					pfVertTemp[4] = ymin;
					pfVertTemp[5] = zmax;

					pfVertTemp[6] = d;
					pfVertTemp[7] = ymax;
					pfVertTemp[8] = zmax;

					//second triangle
					pfVertTemp[9] = d;
					pfVertTemp[10] = ymin;
					pfVertTemp[11] = zmin;

					pfVertTemp[12] = d;
					pfVertTemp[13] = ymax;
					pfVertTemp[14] = zmax;

					pfVertTemp[15] = d;
					pfVertTemp[16] = ymax;
					pfVertTemp[17] = zmin;
				}
				else
				{//ax+y=d
				    //first triangle
					pfVertTemp[0] = xmin;
					pfVertTemp[1] = -a*xmin + d;
					pfVertTemp[2] = zmax;

					pfVertTemp[3] = xmax;
					pfVertTemp[4] = -a*xmax + d;
					pfVertTemp[5] = zmax;

					pfVertTemp[6] = xmax;
					pfVertTemp[7] = -a*xmax + d;
					pfVertTemp[8] = zmin;

					//second triangle
					pfVertTemp[9] = xmin;
					pfVertTemp[10] = -a*xmin + d;
					pfVertTemp[11] = zmin;

					pfVertTemp[12] = xmin;
					pfVertTemp[13] = -a*xmin + d;
					pfVertTemp[14] = zmax;

					pfVertTemp[15] = xmax;
					pfVertTemp[16] = -a*xmax + d;
					pfVertTemp[17] = zmin;
				}
			}
			else
			{//ax+by+cz=d
				//first triangle
				pfVertTemp[0] = xmin;
				pfVertTemp[1] = ymin;
				pfVertTemp[2] = -a*xmin - b*ymin + d;

				pfVertTemp[3] = xmax;
				pfVertTemp[4] = ymin;
				pfVertTemp[5] = -a*xmax - b*ymin + d;

				pfVertTemp[6] = xmax;
				pfVertTemp[7] = ymax;
				pfVertTemp[8] = -a*xmax - b*ymax + d;

				//second triangle
				pfVertTemp[9] = xmin;
				pfVertTemp[10] = ymax;
				pfVertTemp[11] = -a*xmin - b*ymax + d;

				pfVertTemp[12] = xmin;
				pfVertTemp[13] = ymin;
				pfVertTemp[14] = -a*xmin - b*ymin + d;

				pfVertTemp[15] = xmax;
				pfVertTemp[16] = ymax;
				pfVertTemp[17] = -a*xmax - b*ymax + d;
			}
		}
	}
}

int find_neighbor(int index_cur, float * vertices, int iDim, int iNumPoints, int * neighbor_index, float range)
{
	int num_neighbor = 0;

	float * vertex_cur = vertices + index_cur*iDim;
	for (int i = 0; i < iNumPoints; i++)
	{
		bool bNeighbor = true;
		float * vertex_comp = vertices + i*iDim;

		for (int j = 0; j < iDim; j++)
		{
			if (fabsf(vertex_cur[j] - vertex_comp[j]) > range)
			{
				bNeighbor = false;
				break;
			}
		}

		if (bNeighbor)
		{
			neighbor_index[num_neighbor] = i;
			num_neighbor++;
		}
	}

	return num_neighbor;
}

//measure similarity of representative patches of two points
bool similar_rep_patches(struct vertex_cluster * vertex_array, int index_cur, int index_parent, int iDim, float thres_coeff, float thres_dist)
{
	float thres_sim = 0.8;

	int num_rep_cur = vertex_array[index_cur].num_rep;
	float * rep_cur = vertex_array[index_cur].rep_patch_para;
	int num_rep_parent = vertex_array[index_parent].num_rep;
	float * rep_parent = vertex_array[index_parent].rep_patch_para;

	int num_similar = 0;
	for (int cur = 0; cur < num_rep_cur; cur++)
	{
		float * para_cur = rep_cur + cur*(iDim + 1);
		for (int parent = 0; parent < num_rep_parent; parent++)
		{
			float * para_parent = rep_parent + parent*(iDim + 1);
			bool bSimilar = true;
			float sum = 0;
			for (int k = 0; k < iDim; k++)
			{
				if (para_cur[k] * para_parent[k] < 0)
				{
					bSimilar = false;
					break;
				}

				float div = (fabsf(para_cur[k]) > fabsf(para_parent[k])) ? fabsf(para_cur[k]) : fabsf(para_parent[k]);
				if (fabsf((para_cur[k] - para_parent[k]) / (div + 1e-2)) > thres_coeff)//avoid 0/0
				{
					bSimilar = false;
					break;
				}
				sum += para_cur[k] * para_cur[k];
			}
			if (bSimilar)
			{
				float dist = fabsf(para_cur[iDim] - para_parent[iDim]);
				dist = dist / sqrt(sum);
				if (dist < thres_dist)
				{
					num_similar++;
					break;
				}
			}
		}
	}

	int max_rep_num = (num_rep_cur > num_rep_parent) ? num_rep_cur : num_rep_parent;
	if ((num_similar / max_rep_num) > thres_sim)
	{
		return true;
	}
	else
	{
		return false;
	}
}

void search_neighbor(struct vertex_cluster * vertex_array, int index_cur, int index_parent, int cluster, int iDim, float thres_coeff, float thres_dist)
{//Depth First Search
	if (vertex_array[index_cur].cluster_index != -1)
	{
		return;
	}

	if (vertex_array[index_cur].bOutlier)
	{
		return;
	}

	if ((index_cur == index_parent) || similar_rep_patches(vertex_array, index_cur, index_parent, iDim, thres_coeff, thres_dist))
	{
		vertex_array[index_cur].cluster_index = cluster;

		int num_neighbor = vertex_array[index_cur].num_neighbor;
		int * neighbor = vertex_array[index_cur].neighbor;
		for (int i = 0; i < num_neighbor; i++)
		{
			search_neighbor(vertex_array, neighbor[i], index_cur, cluster, iDim, thres_coeff, thres_dist);
		}
	}
}

bool similar_cluster(float * cluster, int index1, int index2, int iDim, float thres_coeff, float thres_dist)
{
	float * para1 = cluster + (iDim + 1)*index1;
	float * para2 = cluster + (iDim + 1)*index2;

	bool bSimilar = true;
	float sum = 0;
	for (int k = 0; k < iDim; k++)
	{
		if (para1[k] * para2[k] < 0)
		{
			bSimilar = false;
			break;
		}

		float div = (fabsf(para1[k]) > fabsf(para2[k])) ? fabsf(para1[k]) : fabsf(para2[k]);
		if (fabsf((para1[k] - para2[k]) / (div + 1e-2)) > thres_coeff)//avoid 0/0
		{
			bSimilar = false;
			break;
		}
		sum += para1[k] * para1[k];
	}
	if (bSimilar)
	{
		float dist = fabsf(para1[iDim] - para2[iDim]);
		dist = dist / sqrt(sum);
		if (dist > thres_dist)
		{
			bSimilar = false;
		}
	}

	return bSimilar;
}

void merge_cluster(struct vertex_cluster * vertex_array, int * cluster_size, int iDim, float * cluster, int iNumPoints, int num_clusters, int * cluster_table, float thres_coeff, float thres_dist)
{
	int * cluster_merge = new int[iNumPoints];
	for (int i = 0; i < iNumPoints; i++)
	{
		cluster_merge[i] = i;
	}

	for (int i = 0; i < num_clusters; i++)
	{
		int cluster_index1 = cluster_table[i];
		if (cluster_size[cluster_index1] == 0)
		{
			continue;
		}

		for (int j = i + 1; j < num_clusters; j++)
		{
			int cluster_index2 = cluster_table[j];
			if (cluster_size[cluster_index2] == 0)
			{
				continue;
			}

			if (similar_cluster(cluster, i, j, iDim, thres_coeff, thres_dist))
			{
				cluster_size[cluster_index1] += cluster_size[cluster_index2];
				cluster_size[cluster_index2] = 0;
				cluster_merge[cluster_index2] = cluster_index1;
			}
		}

	}

	for (int i = 0; i < iNumPoints; i++)
	{
		vertex_array[i].cluster_index = cluster_merge[vertex_array[i].cluster_index];
	}

	delete[] cluster_merge;
}

//calculate line parameters using two points: a*x+b*y=c
void calc_para_lines(float * pfVertChosen, float * pfModel)
{
	float a = 0.0, b = 0.0, c = 0.0;

	float x1 = pfVertChosen[0];
	float y1 = pfVertChosen[1];
	float x2 = pfVertChosen[2];
	float y2 = pfVertChosen[3];
	
	if (fabs(x2 - x1) < 1e-5)
	{
		a = 1.0;
		b = 0.0;
		c = x1;
	}
	else
	{
		float slope = (y2 - y1) / (x2 - x1);
		float intercept = (y1*x2 - y2*x1) / (x2 - x1);
		a = -slope;
		b = 1.0;
		c = intercept;
	}

	*pfModel = a;
	*(pfModel + 1) = b;
	*(pfModel + 2) = c;
}

//calculate plane parameters using three points: a*x+b*y + c*z = d
void calc_para_planes(float * pfVertChosen, float * pfModel)
{
	float a = 0.0, b = 0.0, c = 0.0, d=0.0;

	float x1 = pfVertChosen[0];
	float y1 = pfVertChosen[1];
	float z1 = pfVertChosen[2];
	float x2 = pfVertChosen[3];
	float y2 = pfVertChosen[4];
	float z2 = pfVertChosen[5];
	float x3 = pfVertChosen[6];
	float y3 = pfVertChosen[7];
	float z3 = pfVertChosen[8];

	if ((fabs(x2 - x1) < 1e-5) && (fabs(x3 - x1) < 1e-5))
	{//x=d
		a = 1.0;
		b = 0.0;
		c = 0.0;
		d = x1;
	}
	else if (fabs((y2 - y1) / (x2 - x1) - (y3 - y1) / (x3 - x1)) < 1e-5)
	{//z cannot be represented by a linear combination of x and y
		float slope = (y2 - y1) / (x2 - x1);
		float intercept = (y1*x2 - y2*x1) / (x2 - x1);
		a = -slope;
		b = 1.0;
		c = 0.0;
		d = intercept;
	}
	else
	{
		b = (((z1 - z2)*(x1 - x3) - (z1 - z3)*(x1 - x2)) / ((y1 - y3)*(x1 - x2) - (y1 - y2)*(x1 - x3)));
		a = (- (z1 - z2) - b*(y1 - y2)) / (x1 - x2);
		c = 1.0;
		d = a*x1 + b*y1 + c*z1;
	}

	*pfModel = a;
	*(pfModel + 1) = b;
	*(pfModel + 2) = c;
	*(pfModel + 3) = d;
}

//calculate Euclidean distance
float calc_dist(float * pfCurVert, float * pfModel, int iDim)
{
	float sum1 = 0.0, sum2=0.0;
	for (int i = 0; i < iDim; i++)
	{
		sum1 += pfCurVert[i] * pfModel[i];
		sum2 += pfModel[i] * pfModel[i];
	}
	float dist = fabs(sum1 - pfModel[iDim]) / sqrt(sum2);
	return dist;
}

//using all consensus set to update the line in the manner of least squares 
void least_sqr_2D(float * vertices, float * pfModel, int * cons_set, int cons_set_size)
{
	float sum_x = 0.0, sum_xx = 0.0, sum_y = 0.0, sum_xy = 0.0;
	for (int i = 0; i < cons_set_size; i++)
	{
		float x = vertices[cons_set[i] * 2];
		float y = vertices[cons_set[i] * 2 + 1];
		sum_x += x;
		sum_xx += x*x;
		sum_y += y;
		sum_xy += x*y;
	}

	float denom = cons_set_size*sum_xx - sum_x*sum_x;
	if (fabs(denom) < 1e-5)
	{//x=c
		pfModel[0] = 1.0;
		pfModel[1] = 0.0;
		pfModel[2] = sum_x / (float)cons_set_size;;
	}
	else
	{
		pfModel[0] = -(cons_set_size*sum_xy - sum_x*sum_y) / denom;
		pfModel[1] = 1.0;
		pfModel[2] = (sum_y*sum_xx - sum_xy*sum_x) / denom;
	}

	return;
}

//using all consensus set to update the plane in the manner of least squares 
void least_sqr_3D(float * vertices, float * pfModel, int * cons_set, int cons_set_size)
{
	float sum_x = 0.0, sum_y = 0.0, sum_z = 0.0;
	float sum_xx = 0.0, sum_xy = 0.0, sum_xz = 0.0;
	float sum_yy = 0.0, sum_yz = 0.0, sum_zz = 0.0;

	for (int i = 0; i < cons_set_size; i++)
	{
		float x = vertices[cons_set[i] * 3];
		float y = vertices[cons_set[i] * 3 + 1];
		float z = vertices[cons_set[i] * 3 + 2];
		sum_x += x;
		sum_y += y;
		sum_z += z;
		sum_xx += x*x;
		sum_xy += x*y;
		sum_xz += x*z;
		sum_yy += y*y;
		sum_yz += y*z;
		sum_zz += z*z;
	}

	float m1 = cons_set_size*sum_xy - sum_x*sum_y;
	float m2 = cons_set_size*sum_xz - sum_x*sum_z;
	float m3 = cons_set_size*sum_yz - sum_y*sum_z;
	float m4 = cons_set_size*sum_xx - sum_x*sum_x;
	float m5 = cons_set_size*sum_yy - sum_y*sum_y;
	float m6 = cons_set_size*sum_zz - sum_z*sum_z;

	if (fabs(m4) < 1e-5)
	{//x=d
		pfModel[0] = 1.0;
		pfModel[1] = 0.0;
		pfModel[2] = 0.0;
		pfModel[3] = sum_x / (float)cons_set_size;
	}
	else if (fabs(m4*m5 - m1*m1) < 1e-5)
	{//y=kx+b
		pfModel[0] = - m1 / m4;
		pfModel[1] = 1.0;
		pfModel[2] = 0.0;
		pfModel[3] = (pfModel[0] * sum_x + sum_y) / cons_set_size;
	}
	else
	{
		pfModel[1] = (m1*m2 - m3*m4) / (m4*m5 - m1*m1);//b
		pfModel[2] = 1.0;//c
		pfModel[0] = -(m2 + pfModel[1] * m1) / m4;//a
		pfModel[3] = (pfModel[0] * sum_x + pfModel[1] * sum_y + sum_z) / cons_set_size;//d
	}

	return;
}

void least_squares(float * vertices, float * pfModel, int * cons_set, int cons_set_size, int iDim)
{
	if (iDim == 2)
	{
		least_sqr_2D(vertices, pfModel, cons_set, cons_set_size);
	}
	else if (iDim == 3)
	{
		least_sqr_3D(vertices, pfModel, cons_set, cons_set_size);
	}

	model_rectify(pfModel, 1, iDim);
}

//generate initial patches in the manner of RANSAC
int gen_init_patches(float * vertices, int iDim, int iNumPoints, float * models_para, int iNumModelsMax)
{
	//models initialization
	float * probe = models_para;
	for (int i = 0; i < (iDim+1)*iNumModelsMax; i++, probe++)
	{
		*probe = 0.0;
	}

	float * pfVertChosen = new float[iDim*iDim];
	int * piIndexRand = new int[iDim];
	int * cons_set = new int[iNumPoints];//the set of all the points within a certain range of this patch

	//find the consensus set of this line
	float err_tlr = 1e-2, valid_thres = iNumPoints*0.5;

	int iMaxIte = iNumModelsMax * 50;
	int iModelIndex = 0;//indicate the number of valid patches
	float * pfModelTemp = models_para;
	for (int i = 0; i < iMaxIte; i++)
	{
		//randomly choose iDim vertices to instantiate a model	
		for (int vertNum = 0; vertNum < iDim; vertNum++)
		{
			int iRandTemp = rand() % iNumPoints;//randomly choose a point
			for (int k = 0; k < vertNum; k++)
			{
				if (piIndexRand[k] == iRandTemp)//make sure one point won't be chosen twice
				{
					iRandTemp = rand() % iNumPoints;
					k = -1;
				}
			}
			piIndexRand[vertNum] = iRandTemp;
			for (int j = 0; j < iDim; j++)
			{
				pfVertChosen[vertNum*iDim + j] = vertices[iRandTemp*iDim + j];
			}
		}

		//calculate parameters
		if (iDim == 2)
		{
			calc_para_lines(pfVertChosen, pfModelTemp);
		}
		else if (iDim == 3)
		{
			calc_para_planes(pfVertChosen, pfModelTemp);
		}

		int cons_set_size = 0;
		for (int iVertIndex = 0; iVertIndex < iNumPoints; iVertIndex++)
		{
			float *pfCurVert = vertices + iVertIndex*iDim;
			float dist = calc_dist(pfCurVert, pfModelTemp, iDim);
			if (dist <= err_tlr)
			{
				cons_set[cons_set_size] = iVertIndex;
				cons_set_size++;
			}
		}

		if (cons_set_size >= valid_thres)
		{
			//compute a new model based on the consensus set
			least_squares(vertices, pfModelTemp, cons_set, cons_set_size, iDim);

			iModelIndex++;
			if (iModelIndex >= iNumModelsMax)
			{
				break;
			}
			pfModelTemp += iDim + 1;
		}

	}

	delete[] pfVertChosen;
	delete[] piIndexRand;
	delete[] cons_set;

	return iModelIndex;
}

//standard normal inverse cumulative distribution function
double RationalApproximation(double t)
{
	// Abramowitz and Stegun formula 26.2.23.
	// The absolute value of the error should be less than 4.5 e-4.
	double c[] = { 2.515517, 0.802853, 0.010328 };
	double d[] = { 1.432788, 0.189269, 0.001308 };
	return t - ((c[2] * t + c[1])*t + c[0]) / (((d[2] * t + d[1])*t + d[0])*t + 1.0);
}

double NormalCDFInverse(double p)
{
	if (p <= 0.0 || p >= 1.0)
	{
		printf("Error in calculating the standard normal inverse cumulative distribution function!\n");
		exit(0);
	}

	if (p < 0.5)
	{
		// F^-1(p) = - G^-1(p)
		return -RationalApproximation(sqrt(-2.0*log(p)));
	}
	else
	{
		// F^-1(p) = G^-1(1-p)
		return RationalApproximation(sqrt(-2.0*log(1 - p)));
	}
}

//iterative Kth ordered scale estimation
float ikose(float * res, int &num_inliers, int k, float E)
{
	double epsilon = 1e-6;
	double scale = 0.0, scale_pre=1.0;
	int n = num_inliers;
	double res_k = res[k - 1];

	while ((fabs(scale - scale_pre) > epsilon) && (k < n))
	{
		scale_pre = scale;
		scale = res_k / NormalCDFInverse(0.5*(1 + (double)k / (double)n));

		n = 0;
		while ((n<num_inliers) && (res[n] / scale)<E)//for now, num_inliers is the number of vertices
		{
			n++;
		}
	}

	num_inliers = n;

	return float(scale);
}

//kernel estimation
void calc_patch_score(struct model * patch)
{
	float h = pow((double)(729.0 / (35.0 * patch->num_inliers)), 0.2)*patch->scale;
	float score_temp = 0.0;
	for (int i = 0; i < patch->num_inliers; i++)
	{
		float r_i = patch->inliers_res[i];
		score_temp += (r_i > h) ? 0 : (0.75*(1 - r_i*r_i / (h*h)));
	}
	patch->score = score_temp / (patch->num_inliers * h * patch->scale);
}

//optimal threshold determination
int find_max(float * array, int len)
{
	float max = 0.0;
	int max_index = -1;
	for (int i = 0; i < len; i++)
	{
		if (array[i] > max)
		{
			max = array[i];
			max_index = i;
		}
	}
	return max_index;
}

int find_min(float * array, int len)
{
	float min = 1000000.0;
	int min_index = -1;
	for (int i = 0; i < len; i++)
	{
		if (array[i] < min)
		{
			min = array[i];
			min_index = i;
		}
	}
	return min_index;
}

int otd_pami(float * scores, int len)
{
	int max_index = find_max(scores, len);
	float score_max = scores[max_index];

	double * scores_gap = new double[len];
	double sum = 0.0;
	for (int i = 0; i < len; i++)
	{
		scores_gap[i] = (double)(score_max - scores[i]);
		sum += scores_gap[i];
	}

	double sum_h = 0.0;
	for (int i = 0; i < len; i++)
	{
		scores_gap[i] = scores_gap[i] / sum;
		sum_h += -scores_gap[i] * log(scores_gap[i]+1e-10);//to ensure that log(0) will not happen
	}
	
	int num_sig = 0;
	for (int i = 0; i < len; i++)
	{
		if ((sum_h + log(scores_gap[i]))<0)
		{
			scores[i] = 1.0;
			num_sig++;
		}
		else
		{
			scores[i] = 0.0;
		}
	}

	delete[] scores_gap;
	return num_sig;
}

int otd_tvcg(float * scores, int len)
{
	int max_index = find_max(scores, len);
	float score_max = scores[max_index];
	int min_index = find_min(scores, len);
	float score_min = scores[min_index];
	int hist_len = 1000;
	double hist_int = 1.0 / hist_len;
	double * hist = new double[hist_len];
	for (int i = 0; i < hist_len; i++)
	{
		hist[i] = 0.0;
	}
	for (int i = 0; i < len; i++)
	{
		scores[i] = (scores[i] - score_min) / (score_max - score_min);
		if (scores[i] >= 1.0)
		{
			hist[hist_len - 1]+=1.0;
		}
		else
		{
			hist[int(scores[i] * hist_len)]+=1.0;
		}		
	}
	for (int i = 0; i < hist_len; i++)
	{
		hist[i] /= (double)len;
	}

	double psum_sig = 1.0, psum_weak = 1.0 - psum_sig;
	double max_h = 0.0;
	float opt_thres = 0.0;
	for (int thres = 1; thres < hist_len; thres ++)
	{
		psum_weak += hist[thres - 1];
		psum_sig = 1.0 - psum_weak;
		double hsum_weak = 0.0;
		for (int i = 0; i < thres; i++)
		{
			hsum_weak += hist[i] * hist[i];
		}
		hsum_weak = -log((hsum_weak / (psum_weak*psum_weak + 1e-10)) + 1e-10);
		
		double hsum_sig = 0.0;
		for (int i = thres; i < hist_len; i++)
		{
			hsum_sig += hist[i] * hist[i];
		}
		hsum_sig = -log((hsum_sig / (psum_sig*psum_sig + 1e-10)) + 1e-10);

		if ((hsum_weak + hsum_sig) > max_h)
		{
			max_h = hsum_weak + hsum_sig;
			opt_thres = (float)(thres / 1000.0);
		}
	}

	int num_sig = 0;
	for (int i = 0; i < len; i++)
	{
		if (scores[i] > opt_thres)
		{
			scores[i] = 1.0;
			num_sig++;
		}
		else
		{
			scores[i] = 0.0;
		}
	}

	delete[] hist;

	return num_sig;
}

int otd(float * scores, int len)
{
	//return otd_pami(scores, len);
	return otd_tvcg(scores, len);
}

//clustering over permutations in the manner of medoidshift
void mtx_multiply(double * A, double * B, double * Mul, int dim)
{
	for (int i = 0; i < dim; i++)
	{
		for (int j = 0; j < dim; j++)
		{
			double sum = 0.0;
			for (int k = 0; k < dim; k++)
			{
				sum += A[i*dim + k] * B[k*dim + j];
			}
			Mul[i*dim + j] = sum;
		} 
	}
}

void medoidshift(int * cluster, double * ktdist, double * phi, int dim)
{
	double * mul = new double[dim*dim];
	mtx_multiply(ktdist, phi, mul, dim);

	for (int i = 0; i < dim; i++)
	{
		cluster[i] = -2;  //-2: unprocessed; -1:under processing; others: cluster number
	}

	for (int patch_index = 0; patch_index < dim; patch_index++)
	{
		if (cluster[patch_index] == -2)
		{
			cluster[patch_index] = -1;

			int medoid = -2;
			int next = patch_index;
			while (next != medoid)
			{
				medoid = next;

				double min = 1e20;
				int min_index = -1;
				for (int j = 0; j < dim; j++)
				{
					if (mul[j*dim + next] < min)
					{
						min = mul[j*dim + next];
						min_index = j;
					}
				}

				if (min < 1e-8)//min==0 in case there is only one patch in a cluster
				{
					medoid = patch_index;
					break;
				}
				else if (cluster[min_index] >= 0)
				{
					medoid = cluster[min_index];
					break;
				}
				else if (cluster[min_index] == -1)
				{
					medoid = min_index;
					break;
				}
				else
				{
					cluster[min_index] = -1;
				}

				next = min_index;
			}

			for (int i = 0; i < dim; i++)
			{
				if (cluster[i] == -1)
				{
					cluster[i] = medoid;
				}
			}
		}
	}

	delete[] mul;
}

void clustering_bruteforce(int * cluster, double * ktdist, int dim)
{
	double thres = 0.35;

	for (int i = 0; i < dim; i++)
	{
		cluster[i] = -2;  //-2: unprocessed; -1:under processing; others: cluster number
	}

	for (int patch_index = 0; patch_index < dim; patch_index++)
	{
		if (cluster[patch_index] == -2)
		{
			cluster[patch_index] = patch_index;

			double * ktdist_cur = ktdist + patch_index*dim;
			for (int i = 0; i < dim; i++)
			{
				if (ktdist_cur[i] < thres)
				{
					cluster[i] = patch_index;
				}
			}
		}
	}
}

void kendall_tau_dist(double * ktdist, float * res_init, int * sig_set, int num_sig, int num_points)
{
	struct point_res * res = new struct point_res[num_sig*num_points];//storing the residue of each vertex to each significant patch

	for (int sig_index = 0; sig_index < num_sig; sig_index++)
	{
		struct point_res * res_cur = res + num_points*sig_index;
		float * res_init_cur = res_init + sig_set[sig_index] * num_points;
		for (int i = 0; i < num_points; i++)
		{
			res_cur[i].point_index = i;
			res_cur[i].res = res_init_cur[i];
		}
		mergesort_with_index(res_cur, num_points);
	}

	bool * bProcessd = new bool[num_points];
	double denom = num_points*(num_points - 1) / 2.0;
	for (int i = 0; i < num_sig; i++)
	{
		struct point_res * res_i = res + i*num_points;

		for (int j = 0; j < i; j++)
		{
			struct point_res * res_j = res + j*num_points;

			for (int k = 0; k < num_points; k++)
			{
				bProcessd[k] = false;
			}

			ktdist[i*num_sig + j] = 0.0;

			for (int k = 0; k < num_points; k++)
			{
				int index_search = res_j[k].point_index;
				int num_discor = 0;
				for (int v = 0; v < num_points; v++)
				{
					if ((!bProcessd[v]))
					{
						if(res_i[v].point_index != index_search)
						{
							num_discor++;
						}
						else
						{
							bProcessd[v] = true;
							break;
						}
					}
				}
				ktdist[i*num_sig + j] += (double)num_discor;
			}
			if (ktdist[i*num_sig + j] > 1)
			{
				int temp = 1;
			}

			ktdist[i*num_sig + j] /= denom;
		}
	}

	for (int i = 0; i < num_sig; i++)
	{
		ktdist[i*num_sig + i] = 0.0;
		for (int j = i + 1; j < num_sig; j++)
		{
			ktdist[i*num_sig + j] = ktdist[j*num_sig + i];
		}
	}

	delete[] res;
	delete[] bProcessd;
}

void calc_phi(double * ktdist, double * phi, int dim)
{
	double bandwidth = 0.1;

	for (int i = 0; i < dim; i++)
	{
		for (int j = 0; j < i; j++)
		{
			double ktdh = ktdist[i*dim + j]/bandwidth;
			phi[i*dim + j] = (ktdh > 1) ? 0.0 : 1.5*ktdh;//Epanechnikov kernel
		}
	}

	for (int i = 0; i < dim; i++)
	{
		phi[i*dim + i] = 0.0;
		for (int j = i + 1; j < dim; j++)
		{
			phi[i*dim + j] = phi[j*dim + i];
		}
	}
}

void clustering_over_perm(struct model * patch_init, float * res_init, int * sig_set, int num_sig, int num_points)
{
	double * ktdist = new double[num_sig*num_sig];
	kendall_tau_dist(ktdist, res_init, sig_set, num_sig, num_points);
	
	double * phi = new double[num_sig*num_sig];
	calc_phi(ktdist, phi, num_sig);

	int * cluster = new int[num_sig];
	//medoidshift(cluster, ktdist, phi, num_sig);
	clustering_bruteforce(cluster, ktdist, num_sig);

	for (int i = 0; i < num_sig; i++)
	{
		patch_init[sig_set[i]].group = cluster[i];
	}

	delete[] ktdist;
	delete[] phi;
	delete[] cluster;
}

int subspace_modeling(float * vertices, int iDim, int iNumPoints, float * patch_para, int iNumModelsMax)
{
	//---------------------------------------generating a set of initial patches in the manner of RANSAC-----------------------------
	float * models_para = new float[(iDim + 1)*iNumModelsMax];//patch parameters
	int iModelNum = gen_init_patches(vertices, iDim, iNumPoints, models_para, iNumModelsMax);

	if (iModelNum == 0)
	{
		return 0;
	}

	struct model * models_init = new struct model[iModelNum];
	for (int i = 0; i < iModelNum; i++)
	{
		models_init[i].para = new float[iDim + 1];
		for (int j = 0; j < iDim + 1; j++)
		{
			models_init[i].para[j] = models_para[i*(iDim + 1) + j];
		}
	}

	delete[] models_para;

	//---------------------------------------------------significant patch determination---------------------------------------------
	//calculating each patch's score
	float * scores = new float[iModelNum];
	float * res_total = new float[iModelNum*iNumPoints];//the residue of each point to each patch
	float * inliers_res_init_sorted = new float[iNumPoints];

	int k = (int)(iNumPoints*0.2);
	float E = 2.5;

	for (int model_index = 0; model_index < iModelNum; model_index++)
	{
		float * inliers_res_init = res_total + model_index*iNumPoints;
		struct model * model_cur = models_init + model_index;
		float * para_cur = model_cur->para;
		for (int j = 0; j < iNumPoints; j++)
		{
			inliers_res_init[j] = calc_dist(vertices + j*iDim, para_cur, iDim);
			inliers_res_init_sorted[j] = inliers_res_init[j];
		}

		mergesort(inliers_res_init_sorted, iNumPoints);

		int n_inliers = iNumPoints;
		float scale_cur = ikose(inliers_res_init_sorted, n_inliers, k, E);
		model_cur->scale = scale_cur;
		model_cur->num_inliers = n_inliers;
		model_cur->inliers_index = new int[n_inliers];
		model_cur->inliers_res = new float[n_inliers];

		int i = 0;
		for (int vertex_index = 0; vertex_index < iNumPoints; vertex_index++)
		{
			if ((inliers_res_init[vertex_index] / scale_cur) < E)
			{
				model_cur->inliers_index[i] = vertex_index;
				model_cur->inliers_res[i] = inliers_res_init[vertex_index];
				i++;
			}
		}

		calc_patch_score(model_cur);

		scores[model_index] = model_cur->score;
	}

	//optimal threshold deterination
	int num_sig = otd(scores, iModelNum);//scores is now indicator of significant patches
	int * index_sig = new int[num_sig];
	int sig_index = 0;
	for (int i = 0; i < iModelNum; i++)
	{
		if (scores[i] > 0.5)
		{
			index_sig[sig_index] = i;
			models_init[i].bSig = true;
			sig_index++;
		}
		else
		{
			models_init[i].bSig = false;
		}
	}

	//clean
	delete[] scores;
	delete[] inliers_res_init_sorted;

	//---------------------------------------------representative patch extraction-------------------------------------------
	clustering_over_perm(models_init, res_total, index_sig, num_sig, iNumPoints);

	float * max = new float[num_sig];
	int * max_index = new int[num_sig];
	int num_rep = 0;
	for (int i = 0; i < num_sig; i++)
	{
		max[i] = 0.0;
		max_index[i] = -1;
	}
	for (int i = 0; i < num_sig; i++)
	{
		int model_index_cur = index_sig[i];
		int group_cur = models_init[model_index_cur].group;
		if (max[group_cur] < models_init[model_index_cur].score)
		{
			if (max_index[group_cur] == -1)
			{
				num_rep++;
			}
			max[group_cur] = models_init[model_index_cur].score;
			max_index[group_cur] = model_index_cur;
		}
	}

	int * index_rep = new int[num_rep];
	int index_rep_ind = 0;
	for (int i = 0; i < num_sig; i++)
	{
		if (max_index[i] != -1)
		{
			index_rep[index_rep_ind] = max_index[i];
			index_rep_ind++;
		}
	}

	//return parameters of representative patches
	for (int i = 0; i < num_rep; i++)
	{
		float * patch_para_cur = patch_para + (iDim + 1)*i;
		for (int j = 0; j < (iDim + 1); j++)
		{
			patch_para_cur[j] = models_init[index_rep[i]].para[j];
		}
	}

	//clean
	delete[] res_total;
	delete[] max;
	delete[] max_index;

	delete[] index_sig;
	delete[] index_rep;

	for (int i = 0; i < iModelNum; i++)
	{
		delete[] models_init[i].para;
		delete[] models_init[i].inliers_index;
		delete[] models_init[i].inliers_res;
	}
	delete[] models_init;



	return num_rep;
}

//quick sort
void swap(float * array, int i, int j)
{
	if (i != j)
	{
		float temp = array[i];
		array[i] = array[j];
		array[j] = temp;
	}
}

int partition(float * array, int left, int right)
{
	int len = right - left + 1;
	int pivot = (rand() % len) + left;
	float q = array[pivot];
	swap(array, pivot, right);
	int boundry = left;
	int cur = left;
	while (cur < right)
	{
		if (array[cur] < q)
		{
			swap(array, cur, boundry);
			boundry++;
		}
		cur++;
	}
	swap(array, boundry, right);
	return boundry;
}

void quicksort_ite(float * array, int left, int right)
{
	if (right <= left)
	{
		return;
	}
	int pivot = partition(array, left, right);
	quicksort_ite(array, left, pivot - 1);
	quicksort_ite(array, pivot + 1, right);
}

void quicksort(float * array, int len)
{
	quicksort_ite(array, 0, len - 1);
}

//quick sort with index
void swap_with_index(struct point_res * array, int i, int j)
{
	if (i != j)
	{
		struct point_res temp = array[i];
		array[i] = array[j];
		array[j] = temp;
	}
}

int partition_with_index(struct point_res * array, int left, int right)
{
	int len = right - left + 1;
	int pivot = (rand() % len) + left;
	float q = array[pivot].res;
	swap_with_index(array, pivot, right);
	int boundry = left;
	int cur = left;
	while (cur < right)
	{
		if (array[cur].res < q)
		{
			swap_with_index(array, cur, boundry);
			boundry++;
		}
		cur++;
	}
	swap_with_index(array, boundry, right);
	return boundry;
}

void quicksort_ite_with_index(struct point_res * array, int left, int right)
{
	if (right <= left)
	{
		return;
	}
	int pivot = partition_with_index(array, left, right);
	quicksort_ite_with_index(array, left, pivot - 1);
	quicksort_ite_with_index(array, pivot + 1, right);
}

void quicksort_with_index(struct point_res * array, int len)
{
	quicksort_ite_with_index(array, 0, len - 1);
}

//merge sort
void merge(float * array, int left, int right, int middle)
{
	int n1 = middle - left + 1;
	float * array1 = new float[n1 + 1];
	for (int i = 0; i < n1; i++)
	{
		array1[i] = array[left + i];
	}
	array1[n1] = 1e10;

	int n2 = right - middle;
	float * array2 = new float[n2 + 1];
	for (int i = 0; i < n2; i++)
	{
		array2[i] = array[middle + i + 1];
	}
	array2[n2] = 1e10;

	int i = 0, j = 0;
	for (int k = left; k <= right; k++)
	{
		if (array1[i] <= array2[j])
		{
			array[k] = array1[i];
			i++;
		}
		else
		{
			array[k] = array2[j];
			j++;
		}
	}

	delete[] array1;
	delete[] array2;
}

void mergesort_ite(float * array, int left, int right)
{
	if (left >= right)
	{
		return;
	}
	int middle = (left + right) / 2;//equal to floor
	mergesort_ite(array, left, middle);
	mergesort_ite(array, middle + 1, right);
	merge(array, left, right, middle);
}

void mergesort(float * array, int len)
{
	mergesort_ite(array, 0, len - 1);
}

//merge sort with index
void merge_with_index(struct point_res * array, int left, int right, int middle)
{
	int n1 = middle - left + 1;
	struct point_res * array1 = new struct point_res[n1 + 1];
	for (int i = 0; i < n1; i++)
	{
		array1[i] = array[left + i];
	}
	array1[n1].res = 1e10;

	int n2 = right - middle;
	struct point_res * array2 = new struct point_res[n2 + 1];
	for (int i = 0; i < n2; i++)
	{
		array2[i] = array[middle + i + 1];
	}
	array2[n2].res = 1e10;

	int i = 0, j = 0;
	for (int k = left; k <= right; k++)
	{
		if (array1[i].res <= array2[j].res)
		{
			array[k] = array1[i];
			i++;
		}
		else
		{
			array[k] = array2[j];
			j++;
		}
	}

	delete[] array1;
	delete[] array2;
}

void mergesort_ite_with_index(struct point_res * array, int left, int right)
{
	if (left >= right)
	{
		return;
	}
	int middle = (left + right) / 2;//equal to floor
	mergesort_ite_with_index(array, left, middle);
	mergesort_ite_with_index(array, middle + 1, right);
	merge_with_index(array, left, right, middle);
}

void mergesort_with_index(struct point_res * array, int len)
{
	mergesort_ite_with_index(array, 0, len - 1);
}
