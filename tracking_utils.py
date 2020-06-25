import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
from scipy.stats import gaussian_kde, entropy
from scipy.special import rel_entr

GAUSSIAN_SIGMA_FOR_KDE_Y = 9000
GAUSSIAN_SIGMA_FOR_KDE_X = 300
NUMBER_OF_SAMPLES_TO_KDE = 300
NUMBER_OF_SAMPLES_TO_KL = 1000
BW_FOR_KDE = 100
MU = 0
SIGMA_X_LOCATION = 3
SIGMA_Y_LOCATION = 1.5
SIGMA_X_VELOCITY = 2
SIGMA_Y_VELOCITY = 0.4


def predictParticles(initial_state_matrix):
    # A = np.identity(n=6)
    # A[0, 4] = 1
    # A[1, 5] = 1
    #
    # new_state_matrix = list()
    # for col n initial_state_matrix.T:
    #     new_state_matrix.append(np.matmul(A, col))
    #
    # new_state_matrix = np.array(new_state_matrix).T

    # another way to achieve this
    new_state_matrix = initial_state_matrix
    new_state_matrix[:2, :] = initial_state_matrix[:2, :] + initial_state_matrix[4:, :]

    ## add noise to x,y,vx,vy locations
    new_state_matrix[:1, :] = new_state_matrix[:1, :] + \
                              np.round(np.random.normal(MU, SIGMA_X_LOCATION, size=(1, 100)))
    new_state_matrix[1:2, :] = new_state_matrix[1:2, :] + \
                               np.round(np.random.normal(MU, SIGMA_Y_LOCATION, size=(1, 100)))
    new_state_matrix[4:5, :] = new_state_matrix[4:5, :] + \
                               np.round(np.random.normal(MU, SIGMA_X_VELOCITY, size=(1, 100)))
    new_state_matrix[5:6, :] = new_state_matrix[5:6, :] + \
                               np.round(np.random.normal(MU, SIGMA_Y_VELOCITY, size=(1, 100)))

    return new_state_matrix.astype(int)


def build_KDE(I, rect_info):
    x, y, half_width, half_height, x_vel, y_vel = rect_info
    # temp_image = I[y - half_height:y + half_height, x - half_width:x + half_width]
    mean = [y, x]
    cov = [[GAUSSIAN_SIGMA_FOR_KDE_Y, 0], [0, GAUSSIAN_SIGMA_FOR_KDE_X]]
    y_samples,x_samples = np.random.multivariate_normal(mean, cov, NUMBER_OF_SAMPLES_TO_KDE).T
    y_samples = np.round(y_samples).astype(np.int)
    y_samples = np.maximum(y_samples,0)
    y_samples = np.minimum(y_samples,I.shape[0]-1)
    x_samples = np.round(x_samples).astype(np.int)
    x_samples = np.maximum(x_samples,0)
    x_samples = np.minimum(x_samples,I.shape[1]-1)
    samples = (y_samples,x_samples)


    '''Code to draw circles for kde samples'''
    # image = np.copy(I)
    # for index in range(samples[0].shape[0]):
    #     image = cv2.circle(image, (samples[1][index], samples[0][index]), 5, (0, 255, 0), 2)
    # # Displaying the image
    # cv2.imshow('sas', image)
    # cv2.waitKey(0)

    # temp_image_stacked = temp_image.reshape((temp_image.shape[0]*temp_image.shape[1],3))
    try:
        # indices_choices = np.random.choice(temp_image_stacked.shape[0], NUMBER_OF_SAMPLES_TO_KDE)
        sampled_colors = I[samples]
        pdf = gaussian_kde(sampled_colors.T, bw_method=BW_FOR_KDE)
        return lambda x: pdf(x.T)
    except ValueError:
        input('GOT HERE!')  # TODO - DELETE OR CHANGE TO PRINT
        return None



def compute_KL_div(pdf_p,pdf_q):
    random_colors = np.random.randint(0,255,(NUMBER_OF_SAMPLES_TO_KL,3))
    p_probabilities = pdf_p(random_colors)
    q_probabilities = pdf_q(random_colors)
    a = rel_entr(p_probabilities,q_probabilities)
    result = np.sum(rel_entr(p_probabilities,q_probabilities))
    return result # TODO - MAYBE NEED TO SWITCH ORDER



def compNormHist(I, s_initial):
    x, y, width, height, x_vel, y_vel = s_initial

    # temp_image = I[x - width:x + width, y - height:y + height]
    temp_image = I[y - height:y + height, x - width:x + width]

    temp_image_R = temp_image[:, :, 0] // 16
    temp_image_G = temp_image[:, :, 1] // 16
    temp_image_B = temp_image[:, :, 2] // 16

    histogram = np.zeros((16, 16, 16))

    for i in range(len(temp_image)):
        for j in range(len(temp_image[0])):
            histogram[temp_image_R[i, j]][temp_image_G[i, j]][temp_image_B[i, j]] += 1

    histogram = histogram.reshape((16**3, 1))
    histogram /= np.sum(histogram)

    return histogram


def compBatDist(p, q):
    return np.exp(20 * np.sum(np.sqrt(p * q)))


def get_slacks_from_weights(weights):
    c = [0 for i in range(len(weights))]
    c[0] = weights[0]
    for i in range(1, len(weights)):
        c[i] = weights[i] + c[i - 1]

    return np.array(c)


def measure(image, particles_list, true_pdf):
    weights = list()
    for column in particles_list.T:
        particle_pdf = build_KDE(I=image,rect_info=column)
        # TODO - DIV HERE IS DANGEROUS, REALLY SMALL NUMBER DIVIDING, NEED TO RECEIVE HERE NUMBER BETWEEN 0 TO EXP(20)
        if particle_pdf is None:
            weights.append(0)
        else:
            weights.append(1/compute_KL_div(particle_pdf, true_pdf))

    weights = np.array(weights)
    weights /= np.sum(weights)

    C = get_slacks_from_weights(weights)
    return C, weights


def sampleParticles(S_prev, C):
    sampled_particles = list()
    for i in range(len(S_prev[1])):
        r = np.random.uniform(0, 1)
        j = np.argmax(C >= r)
        sampled_particles.append(S_prev[:, j])

    return np.array(sampled_particles).T


def get_avg_particle(S, W):
    avg_x, avg_y = 0, 0
    for index, particle in enumerate(S.T):
        avg_x += particle[0] * W[index]
        avg_y += particle[1] * W[index]

    return avg_x, avg_y


def get_max_match_particle(S, W):
    max_match_particle = S.T[np.argmax(W)]
    return max_match_particle[0], max_match_particle[1]


def showParticles(I, S, W, i, ID):
    avg_particle_x, avg_particle_y = get_avg_particle(S, W)
    max_particle_x, max_particle_y = get_max_match_particle(S, W)
    try:
        avg_particle_x, avg_particle_y = int(avg_particle_x), int(avg_particle_y)
    except Exception:
        print(S)
        print(W)


    fig, ax = plt.subplots(1)
    # Display the image
    I[:50,:50] = 0
    img_rgb = cv2.cvtColor(I, cv2.COLOR_BGR2RGB)
    ax.imshow(img_rgb)
    # Create a Rectangle patch
    rect_avg = patches.Rectangle((avg_particle_x - S[2][0], avg_particle_y - S[3][0]), S[2][0] * 2, S[3][0] * 2,
                                 linewidth=1, edgecolor='g',
                                 facecolor='none')
    rect_max = patches.Rectangle((max_particle_x - S[2][0], max_particle_y - S[3][0]), S[2][0] * 2, S[3][0] * 2,
                                 linewidth=1, edgecolor='r',
                                 facecolor='none')
    # Add the patch to the Axes
    ax.add_patch(rect_avg)
    ax.add_patch(rect_max)
    for index,state in enumerate(S.T):
        circle1 = plt.Circle((S[0][index], S[1][index]), W[index]*100, color='r')
        ax.add_artist(circle1)



    ax.set_title(f"{ID}-Frame number = {i}")
    # plt.savefig(plot_name)
    plt.show()
    # plt.close('all')
