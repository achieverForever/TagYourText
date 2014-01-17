#ifndef __DOCUMENT_H__
#define __DOCUMENT_H__

class Document
{
public:
	int length;
	int* words;

public:
	Document();
	Document(Document&);
	Document(int len);
	~Document();
};

#endif