#include "document.h"
#include <iostream>

Document::Document(int len)
{
	this->length = len;
	this->words = new int[len];
}

Document::~Document()
{
	if (this->words)
		delete[] this->words;
	this->words = NULL;
	this->length = 0;
}