import json
import numpy as np
import os


TARGET_NAME = 'Smiling'
SENSITIVE_NAME = 'Male'
parent_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

# path = 'D:\\code\\fedfair\\data\\celeba\\raw_data\\img_align_celeba\\000001.jpg'

def get_metadata():
	f_identities = open(os.path.join(
		'data', 'celeba', 'raw_data', 'identity_CelebA.txt'), 'r')
	identities = f_identities.read().split('\n')

	f_attributes = open(os.path.join(
		'data', 'celeba', 'raw_data', 'list_attr_celeba.txt'), 'r')
	attributes = f_attributes.read().split('\n')

	return identities, attributes


def get_celebrities_and_images(identities):
	all_celebs = {}

	for line in identities:
		info = line.split()
		if len(info) < 2:
			continue
		image, celeb = info[0], info[1]
		if celeb not in all_celebs:
			all_celebs[celeb] = []
		all_celebs[celeb].append(image)

	good_celebs = {c: all_celebs[c] for c in all_celebs if len(all_celebs[c]) >= 30}
	num_pic, num_celeb = 0, 0
	for i in good_celebs:
		num_pic += len(good_celebs[i])
		num_celeb += 1
	print(f"{num_pic} imgs in total, for {num_celeb} celebas.")
	return good_celebs


def _get_celebrities_by_image(identities):
	good_images = {}
	for c in identities:
		images = identities[c]
		for img in images:
			good_images[img] = c
	return good_images


def get_celebrities_and_target(celebrities, attributes, attribute_name=TARGET_NAME):
	col_names = attributes[1]
	col_idx = col_names.split().index(attribute_name)

	celeb_attributes = {}
	good_images = _get_celebrities_by_image(celebrities)

	for line in attributes[2:]:
		info = line.split()
		if len(info) == 0:
			continue

		image = info[0]
		if image not in good_images:
			continue
		
		celeb = good_images[image]
		att = (int(info[1:][col_idx]) + 1) / 2
		
		if celeb not in celeb_attributes:
			celeb_attributes[celeb] = []

		celeb_attributes[celeb].append(att)

	return celeb_attributes


def build_json_format(celebrities, targets, sensitive):
	all_data = {}

	celeb_keys = [c for c in celebrities]
	num_samples = [len(celebrities[c]) for c in celeb_keys]
	data = {c: {'x': celebrities[c], 'y': targets[c], 'A': sensitive[c]} for c in celebrities}

	all_data['users'] = celeb_keys
	all_data['user_data'] = data
	all_data['num_samples'] = num_samples
	return all_data


def write_json(json_data):
	file_name = 'all_data.json'
	dir_path = os.path.join('data', 'celeba', 'split_data', 'natural')

	if not os.path.exists(dir_path):
		os.mkdir(dir_path)

	file_path = os.path.join(dir_path, file_name)

	print('writing {}'.format(file_name))
	with open(file_path, 'w') as outfile:
		json.dump(json_data, outfile)

def celeba_generate(SENSITIVE_NAME = 'Male'):
	TARGET_NAME = 'Smiling'
	identities, attributes = get_metadata()
	celebrities = get_celebrities_and_images(identities)
	targets = get_celebrities_and_target(celebrities, attributes, TARGET_NAME)
	sensitive = get_celebrities_and_target(celebrities, attributes, SENSITIVE_NAME)
	
	json_data = build_json_format(celebrities, targets, sensitive)
	write_json(json_data)

def main():
	identities, attributes = get_metadata()
	celebrities = get_celebrities_and_images(identities)
	targets = get_celebrities_and_target(celebrities, attributes, TARGET_NAME)
	sensitive = get_celebrities_and_target(celebrities, attributes, SENSITIVE_NAME)
	
	json_data = build_json_format(celebrities, targets, sensitive)
	write_json(json_data)


if __name__ == '__main__':
	main()


