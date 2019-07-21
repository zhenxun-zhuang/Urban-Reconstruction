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
	float * inst;                //model instantiations
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

//Function prototypes
//Callback
void gl_Key_Callback(GLFWwindow* window, int key, int scancode, int action, int mode);
void scroll_callback(GLFWwindow* window, double xoffset, double yoffset);
void Do_Movement();
//Others
void inst_models(float * models, int iModelNum, float * pfVertInModel, int iDim);
void calc_patch_score(struct model * patch);
void clustering_over_perm(struct model * patch_init, float * res_init, int * sig_set, int num_sig, int num_points);
void mergesort(float * array, int len);
void mergesort_with_index(struct point_res * array, int len);
float calc_dist(float * pfCurVert, float * pfModel, int iDim);
float ikose(float * res, int &num_inliers, int k, float E);
int    gen_init_patches(float * vertices, int iDim, int iNumPoints, float * models, int iNumModelsMax);
int    otd(float * scores, int len);


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
	for (int i = 0; i < iNumPoints; i ++)
	{
		for (int j = 0; j < iDim-1; j++)
		{
			fscanf_s(fInputData, "%f,", vertices_ind);//suppose input is a csv file
			vertices_ind++;
		}
		fscanf_s(fInputData, "%f", vertices_ind);
		vertices_ind++;
	}
	fclose(fInputData);

	end_time = clock();
	elapsed_min = (end_time - start_time) / (CLOCKS_PER_SEC * 60);
	elapsed_sec = (end_time - elapsed_min * 60 * CLOCKS_PER_SEC - start_time) / CLOCKS_PER_SEC;
	printf("%02ld:%02ld Successfully read the whole point cloud!\n", elapsed_min, elapsed_sec);


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


	//---------------------------------------generating a set of initial patches in the manner of RANSAC-----------------------------
	int iNumModelsMax = 200;//Maximum initial patch numbers
	float * models_para = new float[(iDim + 1)*iNumModelsMax];//patch parameters
	int iModelNum = gen_init_patches(vertices, iDim, iNumPoints, models_para, iNumModelsMax);
	int num_vertices_per_model = (iDim == 2) ? 2 : 6;//number of vertices used for instantiation per model
	float * pfVertInModel = new float[num_vertices_per_model * iDim * iModelNum];//patch instantiations
	inst_models(models_para, iModelNum, pfVertInModel, iDim);

	struct model * models_init = new model[iModelNum];
	for (int i = 0; i < iModelNum; i++)
	{
		models_init[i].para = new float[iDim + 1];
		for (int j = 0; j < iDim + 1; j++)
		{
			models_init[i].para[j] = models_para[i*(iDim + 1) + j];
		}

		models_init[i].inst = new float[num_vertices_per_model*iDim];
		for (int j = 0; j < num_vertices_per_model*iDim; j++)
		{
			models_init[i].inst[j] = pfVertInModel[i*(num_vertices_per_model*iDim) + j];
		}
	}

	end_time = clock();
	elapsed_min = (end_time - start_time) / (CLOCKS_PER_SEC * 60);
	elapsed_sec = (end_time - elapsed_min * 60 * CLOCKS_PER_SEC - start_time) / CLOCKS_PER_SEC;
	printf("%02ld:%02ld Successfully generate %d initalial patches using RANSAC!\n", elapsed_min, elapsed_sec, iModelNum);

	//Generate vertex data buffer
	GLuint VBO_init_patches;
	glGenBuffers(1, &VBO_init_patches);
	//Generate VAO
	GLuint VAO_init_patches;
	glGenVertexArrays(1, &VAO_init_patches);
	//Bind vertex array object
	glBindVertexArray(VAO_init_patches);
	//Bind vertex data buffer
	glBindBuffer(GL_ARRAY_BUFFER, VBO_init_patches);
	//Copy vertex data
	glBufferData(GL_ARRAY_BUFFER, sizeof(GLfloat) * num_vertices_per_model * iDim * iModelNum, pfVertInModel, GL_STATIC_DRAW);
	//Link vertex data with vertex attribute
	glVertexAttribPointer(0, iDim, GL_FLOAT, GL_FALSE, iDim * sizeof(GLfloat), (const void *)0);
	glEnableVertexAttribArray(0);
	//Unbind vertex data buffer
	//Note that this is allowed, the call to glVertexAttribPointer registered VBO as the currently bound vertex buffer object so afterwards we can safely unbind
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	//Unbind vertex array object
	glBindVertexArray(0);

	//---------------------------------------------------significant patch determination---------------------------------------------
	//calculating each patch's score
	float * scores = new float[iModelNum];
	float * res_total = new float[iModelNum*iNumPoints];//the residue of each point to each patch
	float * inliers_res_init_sorted = new float[iNumPoints];

	int k = (int)(iNumPoints*0.01);
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
	float * inst_sig = new float[num_sig*num_vertices_per_model*iDim];//instantiations of significant patches
	int * index_sig = new int[num_sig];
	int sig_index = 0;
	for (int i = 0; i < iModelNum; i++)
	{
		if (scores[i] > 0.5)
		{
			index_sig[sig_index] = i;
			models_init[i].bSig = true;
			for (int j = 0; j < num_vertices_per_model*iDim; j++)
			{
				inst_sig[sig_index*num_vertices_per_model*iDim + j] = pfVertInModel[i*num_vertices_per_model*iDim + j];
			}
			sig_index++;
		}
		else
		{
			models_init[i].bSig = false;
		}
	}

	end_time = clock();
	elapsed_min = (end_time - start_time) / (CLOCKS_PER_SEC * 60);
	elapsed_sec = (end_time - elapsed_min * 60 * CLOCKS_PER_SEC - start_time) / CLOCKS_PER_SEC;
	printf("%02ld:%02ld Successfully determine %d significant patches!\n", elapsed_min, elapsed_sec, num_sig);

	//Generate vertex data buffer
	GLuint VBO_sig_patches;
	glGenBuffers(1, &VBO_sig_patches);
	//Generate VAO
	GLuint VAO_sig_patches;
	glGenVertexArrays(1, &VAO_sig_patches);
	//Bind vertex array object
	glBindVertexArray(VAO_sig_patches);
	//Bind vertex data buffer
	glBindBuffer(GL_ARRAY_BUFFER, VBO_sig_patches);
	//Copy vertex data
	glBufferData(GL_ARRAY_BUFFER, sizeof(GLfloat) * num_vertices_per_model * iDim * num_sig, inst_sig, GL_STATIC_DRAW);
	//Link vertex data with vertex attribute
	glVertexAttribPointer(0, iDim, GL_FLOAT, GL_FALSE, iDim * sizeof(GLfloat), (const void *)0);
	glEnableVertexAttribArray(0);
	//Unbind vertex data buffer
	//Note that this is allowed, the call to glVertexAttribPointer registered VBO as the currently bound vertex buffer object so afterwards we can safely unbind
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	//Unbind vertex array object
	glBindVertexArray(0);

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


	float * inst_rep = new float[num_vertices_per_model*iDim*num_rep];//instantiations of significant patches
	for (int i = 0; i < num_rep; i++)
	{
		float * inst_rep_cur = inst_rep + num_vertices_per_model*iDim*i;
		float * inst_model_cur = models_init[index_rep[i]].inst;
		for (int j = 0; j < num_vertices_per_model*iDim; j++)
		{
			inst_rep_cur[j] = inst_model_cur[j];
		}
	}

	end_time = clock();
	elapsed_min = (end_time - start_time) / (CLOCKS_PER_SEC * 60);
	elapsed_sec = (end_time - elapsed_min * 60 * CLOCKS_PER_SEC - start_time) / CLOCKS_PER_SEC;
	printf("%02ld:%02ld Successfully cluster %d significant patches into %d groups!\n", elapsed_min, elapsed_sec, num_sig, num_rep);

	//Generate vertex data buffer
	GLuint VBO_rep_patches;
	glGenBuffers(1, &VBO_rep_patches);
	//Generate VAO
	GLuint VAO_rep_patches;
	glGenVertexArrays(1, &VAO_rep_patches);
	//Bind vertex array object
	glBindVertexArray(VAO_rep_patches);
	//Bind vertex data buffer
	glBindBuffer(GL_ARRAY_BUFFER, VBO_rep_patches);
	//Copy vertex data
	glBufferData(GL_ARRAY_BUFFER, sizeof(GLfloat) * num_vertices_per_model * iDim * num_rep, inst_rep, GL_STATIC_DRAW);
	//Link vertex data with vertex attribute
	glVertexAttribPointer(0, iDim, GL_FLOAT, GL_FALSE, iDim * sizeof(GLfloat), (const void *)0);
	glEnableVertexAttribArray(0);
	//Unbind vertex data buffer
	//Note that this is allowed, the call to glVertexAttribPointer registered VBO as the currently bound vertex buffer object so afterwards we can safely unbind
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	//Unbind vertex array object
	glBindVertexArray(0);


	float ** inst_cluster = new float*[num_rep];
	int * size_cluster = new int[num_sig];
	int * index_table = new int[num_sig];//indicate cluster index of each group
	for (int i = 0; i < num_sig; i++)
	{
		size_cluster[i] = 0;
		index_table[i] = -1;
	}
	int index_in_table = 0;
	for (int i = 0; i < num_sig; i++)
	{
		int group_cur = models_init[index_sig[i]].group;
		size_cluster[group_cur]++;
		if (index_table[group_cur] == -1)
		{
			index_table[group_cur] = index_in_table;
			index_in_table++;
		}
	}

	for (int i = 0; i < num_sig; i++)
	{
		if (size_cluster[i] > 0)
		{
			inst_cluster[index_table[i]] = new float[num_vertices_per_model*iDim*size_cluster[i]];
		}
	}

	int * inst_cluster_index = new int[num_rep];
	for (int i = 0; i < num_rep; i++)
	{
		inst_cluster_index[i] = 0;
	}

	for (int i = 0; i < num_sig; i++)
	{
		int group_cur = models_init[index_sig[i]].group;
		int group_in_table = index_table[group_cur];//the cluster index
		float * inst_in_patch = models_init[index_sig[i]].inst;
		float * inst_in_cluster_cur = inst_cluster[group_in_table] + inst_cluster_index[group_in_table];
		for (int j = 0; j < num_vertices_per_model*iDim; j++)
		{
			inst_in_cluster_cur[j] = inst_in_patch[j];
		}
		inst_cluster_index[group_in_table] += num_vertices_per_model*iDim;
	}

	end_time = clock();
	elapsed_min = (end_time - start_time) / (CLOCKS_PER_SEC * 60);
	elapsed_sec = (end_time - elapsed_min * 60 * CLOCKS_PER_SEC - start_time) / CLOCKS_PER_SEC;
	printf("%02ld:%02ld Successfully find %d representative patches:\n", elapsed_min, elapsed_sec, num_rep);

	for (int i = 0; i < num_rep; i++)
	{
		struct model * model_cur = models_init + index_rep[i];
		printf("      %d:   ", i);
		if (iDim == 2)
		{
			if (model_cur->para[1] < 0.5)
			{//x=c
				printf("x = %.4f\n", model_cur->para[2]);
			}
			else
			{//y=kx+b
				printf("y = %.4f*x + %.4f\n", -model_cur->para[0], model_cur->para[2]);
			}
		}
		else if (iDim == 3)
		{
			if (model_cur->para[2] < 0.5)
			{
				if (model_cur->para[1] < 0.5)
				{//x=d
					printf("x = %.4f\n", model_cur->para[3]);
				}
				else
				{//ax+y=d
					printf("%.4fx + y = %.4f\n", model_cur->para[0], model_cur->para[3]);
				}
			}
			else
			{//ax+by+cz=d
				printf("%.4fx + %.4fy + z = %.4f\n", model_cur->para[0], model_cur->para[1], model_cur->para[3]);
			}
		}
	}

	GLuint * VBO_cluster_patches = new GLuint[num_rep];
	glGenBuffers(num_rep, VBO_cluster_patches);
	GLuint * VAO_cluster_patches = new GLuint[num_rep];
	glGenVertexArrays(num_rep, VAO_cluster_patches);
	for (int i = 0; i < num_rep; i++)
	{
		//Bind vertex array object
		glBindVertexArray(VAO_cluster_patches[i]);
		//Bind vertex data buffer
		glBindBuffer(GL_ARRAY_BUFFER, VBO_cluster_patches[i]);
		//Copy vertex data
		glBufferData(GL_ARRAY_BUFFER, sizeof(GLfloat) * inst_cluster_index[i], inst_cluster[i], GL_STATIC_DRAW);
		//Link vertex data with vertex attribute
		glVertexAttribPointer(0, iDim, GL_FLOAT, GL_FALSE, iDim * sizeof(GLfloat), (const void *)0);
		glEnableVertexAttribArray(0);
		//Unbind vertex data buffer
		//Note that this is allowed, the call to glVertexAttribPointer registered VBO as the currently bound vertex buffer object so afterwards we can safely unbind
		glBindBuffer(GL_ARRAY_BUFFER, 0);
		//Unbind vertex array object
		glBindVertexArray(0);
	}


	delete[] vertices;

	delete[] models_para;
	delete[] pfVertInModel;

	for (int i = 0; i < iModelNum; i++)
	{
		delete[] models_init[i].para;
		delete[] models_init[i].inst;
		delete[] models_init[i].inliers_index;
		delete[] models_init[i].inliers_res;
	}
	delete[] models_init;

	delete[] index_sig;
	delete[] inst_sig;

	delete[] res_total;

	delete[] max;
	delete[] max_index;

	delete[] size_cluster;
	delete[] index_table;

	for (int i = 0; i < num_rep; i++)
	{
		delete[] inst_cluster[i];
	}
	if (num_rep > 0)
	{
		delete[] inst_cluster;
		delete[] index_rep;
		delete[] inst_rep;
	}


	//--------------------------------------------------show time-----------------------------------------------------------------------
	printf("-------------------------------------------------------------------------------------------------\n");
	printf("Press following buttons to show results of each step:\n");
	printf("0 --- Point Cloud\n");
	printf("1 --- Initial Patches\n");
	printf("2 --- Significant patches\n");
	printf("3 --- Clusters\n");
	printf("4 --- Representative Patches\n");
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
				glUniform4f(vertexColorLocation, 1.0f, 1.0f, 1.0f, 1.0f);
				glBindVertexArray(VAO_pointCloud);
				glDrawArrays(GL_POINTS, 0, iNumPoints);
			}
			if (state == 1)
			{
				glUniform4f(vertexColorLocation, 1.0f, 0.0f, 0.0f, 0.0f);
				glBindVertexArray(VAO_init_patches);
				glDrawArrays(GL_LINES, 0, num_vertices_per_model * iModelNum);
			}
			if (state == 2)
			{
				glUniform4f(vertexColorLocation, 0.0f, 1.0f, 0.0f, 0.0f);
				glBindVertexArray(VAO_sig_patches);
				glDrawArrays(GL_LINES, 0, num_vertices_per_model * num_sig);
			}
			if (state == 3)
			{
				for (int i = 0; i < num_rep; i++)
				{
					glUniform4f(vertexColorLocation, i / (float)num_rep, cosf(float(i)), 1.0f - 0.5*i / (float)num_rep, 0.0f);
					glBindVertexArray(VAO_cluster_patches[i]);
					glDrawArrays(GL_LINES, 0, inst_cluster_index[i] / iDim);
				}
			}
			if (state == 4)
			{
				glUniform4f(vertexColorLocation, 1.0f, 0.95556f, 0.0f, 0.0f);
				glBindVertexArray(VAO_rep_patches);
				glDrawArrays(GL_LINES, 0, num_vertices_per_model * num_rep);
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
				glUniform4f(vertexColorLocation, 1.0f, 0.0f, 0.0f, 0.0f);
				glBindVertexArray(VAO_init_patches);
				glDrawArrays(GL_TRIANGLES, 0, num_vertices_per_model * iModelNum);
			}
			if (state == 2)
			{
				glUniform4f(vertexColorLocation, 0.0f, 1.0f, 0.0f, 0.0f);
				glBindVertexArray(VAO_sig_patches);
				glDrawArrays(GL_TRIANGLES, 0, num_vertices_per_model * num_sig);
			}
			if (state == 3)
			{
				for (int i = 0; i < num_rep; i++)
				{
					glUniform4f(vertexColorLocation, i / (float)num_rep, cosf(float(i)), 1.0f - 0.5*i / (float)num_rep, 0.0f);
					glBindVertexArray(VAO_cluster_patches[i]);
					glDrawArrays(GL_TRIANGLES, 0, inst_cluster_index[i] / iDim);
				}
			}
			if (state == 4)
			{
				glBindVertexArray(VAO_rep_patches);
				for (int i = 0; i < num_rep; i++)
				{
					glUniform4f(vertexColorLocation, i / (float)num_rep, cosf(float(i)), 1.0f - 0.5*i / (float)num_rep, 0.0f);
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
	glDeleteVertexArrays(1, &VAO_init_patches);
	glDeleteBuffers(1, &VBO_init_patches);
	glDeleteVertexArrays(1, &VAO_sig_patches);
	glDeleteBuffers(1, &VBO_sig_patches);
	glDeleteVertexArrays(1, &VAO_rep_patches);
	glDeleteBuffers(1, &VBO_rep_patches);
	glDeleteVertexArrays(num_rep, VAO_cluster_patches);
	glDeleteBuffers(num_rep, VBO_cluster_patches);

	//The end
	glfwTerminate();

	printf("Thank you for using this platform!\n");
	printf("-------------------------------------------------------------------------------------------------\n");

	if (num_rep > 0)
	{
		delete[] inst_cluster_index;
		delete[] VBO_cluster_patches;
		delete[] VAO_cluster_patches;
	}

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
	float err_tlr = 8e-3, valid_thres = iNumPoints*0.015;

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
			if (iDim == 2)
			{
				least_sqr_2D(vertices, pfModelTemp, cons_set, cons_set_size);
			}
			else if (iDim == 3)
			{
				least_sqr_3D(vertices, pfModelTemp, cons_set, cons_set_size);
			}

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

//calculate points falling in this patch
void inst_models(float * models_para, int iModelNum, float * pfVertInModel, int iDim)
{
	int num_vertices_per_model = (iDim == 2) ? 2 : 6;
	if (iDim == 2)//lines
	{
		for (int i = 0; i < iModelNum; i++)
		{
			float * pfModelTemp = models_para + i*(iDim + 1);
			float a = *pfModelTemp;
			float b = *(pfModelTemp + 1);
			float c = *(pfModelTemp + 2);

			float * pfVertTemp = pfVertInModel + num_vertices_per_model * iDim * i;
			if (fabs(b) < 1e-5)//x=c
			{
				*pfVertTemp = c;
				*(pfVertTemp + 1) = -1.0;
				*(pfVertTemp + 2) = c;
				*(pfVertTemp + 3) = 1.0;
			}
			else
			{
				*pfVertTemp = -1.0;
				*(pfVertTemp + 1) = a + c;
				*(pfVertTemp + 2) = 1.0;
				*(pfVertTemp + 3) = -a + c;
			}
		}
	}
	else if (iDim == 3)//planes
	{
		for (int i = 0; i < iModelNum; i++)
		{
			float * pfModelTemp = models_para + i*(iDim + 1);
			float a = *pfModelTemp;
			float b = *(pfModelTemp + 1);
			float c = *(pfModelTemp + 2);
			float d = *(pfModelTemp + 3);

			float * pfVertTemp = pfVertInModel + num_vertices_per_model * iDim * i;
			if (fabs(c) < 1e-5)
			{
				if (fabs(b) < 1e-5)
				{//x=d
					//first triangle
					pfVertTemp[0] = d;
					pfVertTemp[1] = -1.0;
					pfVertTemp[2] = -1.0;

					pfVertTemp[3] = d;
					pfVertTemp[4] = -1.0;
					pfVertTemp[5] = 1.0;

					pfVertTemp[6] = d;
					pfVertTemp[7] = 1.0;
					pfVertTemp[8] = 1.0;

					//second triangle
					pfVertTemp[9] = d;
					pfVertTemp[10] = -1.0;
					pfVertTemp[11] = -1.0;

					pfVertTemp[12] = d;
					pfVertTemp[13] = 1.0;
					pfVertTemp[14] = 1.0;

					pfVertTemp[15] = d;
					pfVertTemp[16] = 1.0;
					pfVertTemp[17] = -1.0;
				}
				else
				{//ax+y=d
					//first triangle
					pfVertTemp[0] = -1.0;
					pfVertTemp[1] = a + d;
					pfVertTemp[2] = 1.0;

					pfVertTemp[3] = 1.0;
					pfVertTemp[4] = -a + d;
					pfVertTemp[5] = 1.0;

					pfVertTemp[6] = 1.0;
					pfVertTemp[7] = -a + d;
					pfVertTemp[8] = -1.0;

					//second triangle
					pfVertTemp[9] = -1.0;
					pfVertTemp[10] = a + d;
					pfVertTemp[11] = -1.0;

					pfVertTemp[12] = -1.0;
					pfVertTemp[13] = a + d;
					pfVertTemp[14] = 1.0;

					pfVertTemp[15] = 1.0;
					pfVertTemp[16] = -a + d;
					pfVertTemp[17] = -1.0;
				}
			}
			else
			{
				//first triangle
				pfVertTemp[0] = -1.0;
				pfVertTemp[1] = -1.0;
				pfVertTemp[2] = a + b +d;

				pfVertTemp[3] = 1.0;
				pfVertTemp[4] = -1.0;
				pfVertTemp[5] = -a + b + d;

				pfVertTemp[6] = 1.0;
				pfVertTemp[7] = 1.0;
				pfVertTemp[8] = -a - b + d;

				//second triangle
				pfVertTemp[9] = -1.0;
				pfVertTemp[10] = 1.0;
				pfVertTemp[11] = a - b + d;

				pfVertTemp[12] = -1.0;
				pfVertTemp[13] = -1.0;
				pfVertTemp[14] = a + b + d;

				pfVertTemp[15] = 1.0;
				pfVertTemp[16] = 1.0;
				pfVertTemp[17] = -a - b + d;
			}
		}
	}
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
	float max_h = 0.0;
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

#if DEBUG_ZZX
	FILE * fpzzx = NULL;
	errno_t err = fopen_s(&fpzzx, "debug_zzx.txt", "a");

	for (int i = 0; i <dim; i++)
	{
		for (int j = 0; j < dim; j++)
		{
			fprintf(fpzzx, "%.3f   ", ktdist[i*dim + j]);
		}
		fprintf(fpzzx, "\n");
	}
	fprintf(fpzzx, "\n\n");

	for (int i = 0; i <dim; i++)
	{
		for (int j = 0; j < dim; j++)
		{
			fprintf(fpzzx, "%.3f   ", phi[i*dim + j]);
		}
		fprintf(fpzzx, "\n");
	}
	fprintf(fpzzx, "\n\n");

	for (int i = 0; i <dim; i++)
	{
		for (int j = 0; j < dim; j++)
		{
			fprintf(fpzzx, "%.7f   ", mul[i*dim + j]);
		}
		fprintf(fpzzx, "\n");
	}
	fprintf(fpzzx, "\n");

	fclose(fpzzx);
#endif

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

				double min = 1e10;
				int min_index = -1;
				for (int j = 0; j < dim; j++)
				{
					if (mul[j*dim + next] < min)
					{
						min = mul[j*dim + next];
						min_index = j;
					}
				}

				if (min < 1e-5)//min==0 in case there is only one patch in a cluster
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
	double thres = 0.1;

	for (int i = 0; i < dim; i++)
	{
		cluster[i] = -2;  //-2: unprocessed;  others: cluster number
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
	printf("Calculating KD dist: ");
	for (int i = 0; i < num_sig; i++)
	{
		if (i > 1)
		{
			printf("\b\b\b\b\b\b\b");
		}
		if (i > 0)
		{
			printf("%03d ", i);
		}

		struct point_res * res_i = res + i*num_points;

		for (int j = 0; j < i; j++)
		{
			if (j > 0)
			{
				printf("\b\b\b");
			}
			printf("%03d", j);

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
			ktdist[i*num_sig + j] /= denom;
		}
	}

	for (int bn = 0; bn < 28; bn++)
	{
		printf("\b");
	}
	printf("\n");

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
	float * array1 = new float[n1+1];
	for (int i = 0; i < n1; i++)
	{
		array1[i] = array[left + i];
	}
	array1[n1] = 1e10;

	int n2 = right - middle;
	float * array2 = new float[n2 + 1];
	for (int i = 0; i < n2; i++)
	{
		array2[i] = array[middle + i +1];
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
