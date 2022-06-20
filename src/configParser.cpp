#include "IniManager.h"
#ifndef __linux__
#include <io.h>
#else
#include <unistd.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#endif
#include <algorithm>
#include <functional> 
#include <cctype>
#include <locale>

// trim from start
static inline std::string &ltrim(std::string &s) {
	s.erase(s.begin(), std::find_if(s.begin(), s.end(), std::not1(std::ptr_fun<int, int>(std::isspace))));
	return s;
}
// trim from end
static inline std::string &rtrim(std::string &s) {
	s.erase(std::find_if(s.rbegin(), s.rend(), std::not1(std::ptr_fun<int, int>(std::isspace))).base(), s.end());
	return s;
}
// trim from both ends
static inline std::string &trim(std::string &s) {
	return ltrim(rtrim(s));
}

void IniManager::ClearSection(const char* szSection)
{
	IniFile::iterator it;
	if ((it = m_mapValues.find(szSection)) != m_mapValues.end())
	{
		it->second.clear();
	}
}

// return pointer to the rest of buffer, NULL if last line
// current line in buf is zeroed
char* IniGetLine(char* buf)
{
	char* ptr;
	ptr = strchr(buf, '\n');
	if (ptr)
		*ptr++ = 0;
	return ptr;
}

// return type of line: 0 - commented line; 1 - new section; 2 - (possibly) value
int IniGetLineType(char* str)
{
	if (str[0] == ';')
		return 0;
	if (str[0] == '[')
		return 1;
	return 2;
}

// Get section name (points into str), returns NULL if name cannot be obtained; ']' and trailing spaces will be zeroed
char* IniGetSectionName(char* str)
{
	if (str[0] != '[')
		return NULL;
	str++;
	while (*str == ' ')
		str++;
	char* tmp = strchr(str, ']');
	if (!tmp)
		return NULL;
	do
	{
		*tmp-- = 0;
	} while (*tmp == ' ');
	return str;
}

void IniManager::ReadIniFile()
{
	FILE* fp;
	char* buf;
	unsigned len;
	char * str, *ptr;
	IniSection mapData;
	std::string sSection;
	fp = fopen(m_sFileName.c_str(), "rt");
	if (!fp)
		return;

#ifndef __linux__
	buf = (char*)malloc(len = 1 + _filelength(_fileno(fp)));
#else
	int prev=ftell(fp);
	fseek(fp, 0L, SEEK_END);
	int sz=ftell(fp);
	fseek(fp,prev,SEEK_SET);
	buf = (char*)malloc(len = 1 + sz);
#endif

	if (!buf)
		goto end;
	len = fread(buf, 1, len - 1, fp);
	if (!len)
		goto end;
	buf[len] = 0;
	ptr = buf;
	do
	{
		str = ptr;
		ptr = IniGetLine(str);
		switch (IniGetLineType(str))
		{
		case 1:
		{
			if (!sSection.empty())
			{
				m_mapValues[trim(sSection)] = mapData;
			}
			// new section
			mapData.clear();
			str = IniGetSectionName(str);
			if (str)
			{
				sSection = str;
				transform(sSection.begin(), sSection.end(), sSection.begin(), ::tolower);
			}
		}
		break;
		case 2:
		{
			std::string name;
			// check that in section, otherwise no need to process
			if (sSection.empty())
				break;
			char * data;
			data = strchr(str, '=');
			if (data)
			{
				*data++ = 0;
				name = str;
				std::transform(name.begin(), name.end(), name.begin(), ::tolower);

#ifndef __linux__
				mapData[trim(name)] = trim(std::string(data));
#else
				std::string tmpStr(data);
				trim((std::string&)tmpStr);
				
				mapData[trim(name)] = tmpStr;
#endif

			}
		}
		break;
		case 0:
		default:
			break;
		}
	} while (ptr);

	if (!sSection.empty())
		m_mapValues[trim(sSection)] = mapData;

end:
	if (buf)
		free(buf);
	if (fp)
		fclose(fp);
}
/*
void IniManager::ReadIniFile()
{
const unsigned nBufSize = 0x7FFF;
char * szSectBuf = new char[nBufSize];
char * szKeyBuf = new char[nBufSize];
//KVD   char * szValBuf = new char[nBufSize];
//KVD   char * szDefault = "";
char * pSectPtr, * pKeyPtr, * pValPtr;
unsigned nRead = 0;
IniSection mapEmpty; // empty map

m_mapValues.clear();
//KVD   nRead = GetPrivateProfileString(NULL, NULL, szDefault, szSectBuf, nBufSize, m_sFileName.c_str());
// get section names (zeroed strings in buffer)
nRead = GetPrivateProfileSectionNames(szSectBuf, nBufSize, m_sFileName.c_str());
if(nRead)
{
pSectPtr = szSectBuf;
while(*pSectPtr)
{
IniFile::iterator it =
m_mapValues.insert(pair<string, IniSection >(pSectPtr, mapEmpty)).first;
ASSERT(it != m_mapValues.end());
//KVD       nRead = GetPrivateProfileString(pSectPtr, NULL, szDefault, szKeyBuf, nBufSize, m_sFileName.c_str());
// get keys and values for current section (zeroed strings with format 'key=value')
nRead = GetPrivateProfileSection(pSectPtr, szKeyBuf, nBufSize, m_sFileName.c_str());
if(nRead)
{
pKeyPtr = szKeyBuf;
while(*pKeyPtr)
{
//KVD           nRead = GetPrivateProfileString(pSectPtr, pKeyPtr, szDefault, szValBuf, nBufSize, m_sFileName.c_str());
//KVD           it->second[pKeyPtr] = szValBuf;
nRead = strlen(pKeyPtr);
pValPtr = strchr(pKeyPtr, '=');
if(pValPtr)
{
*pValPtr++ = 0;
//KVD             string key = pKeyPtr;
//KVD             string val = pValPtr;
//KVD             key = key.substr(0, pValPtr-pKeyPtr-1);
it->second[pKeyPtr] = pValPtr;
}
pKeyPtr += nRead + 1;
}
}
pSectPtr += strlen(pSectPtr) + 1;
}
}
delete [] szSectBuf;
delete [] szKeyBuf;
//KVD   delete [] szValBuf;
}*/

/*
void IniManager::WriteIniFile()
{
BOOL bWritten = TRUE;
char* buf = new char[0x7FFF];
char* ptr;
DeleteFile(m_sFileName.c_str()); //!!! is it REALLY necessary???
IniFile::const_iterator mapit = m_mapValues.begin();
for(; mapit != m_mapValues.end(); mapit++)
{
ptr = buf;
IniSection::const_iterator it = mapit->second.begin();
for(; it != mapit->second.end(); it++)
{
//KVD       WritePrivateProfileString( mapit->first.c_str(), it->first.c_str(), it->second.c_str(), m_sFileName.c_str());
sprintf(ptr, "%s=%s", it->first.c_str(), it->second.c_str());
ptr += strlen(ptr)+1;
}
*ptr = 0;
WritePrivateProfileSection(mapit->first.c_str(), buf, m_sFileName.c_str());
}
delete [] buf;
}
*/
void IniManager::Write(std::ostream& to) const
{
	IniFile::const_iterator mapit;
	for (mapit = m_mapValues.begin(); mapit != m_mapValues.end(); mapit++) {
		to << "[" << mapit->first << "]" << std::endl;
		IniSection::const_iterator it;
		for (it = mapit->second.begin(); it != mapit->second.end(); it++) {
			to << it->first << "=" << it->second << std::endl;
		}
	}
}

void IniManager::Copy(const IniManager& from, bool bCopyValues, bool bClear)
{
	if (bClear)
		m_mapValues.clear();
	IniFile::const_iterator mapit = from.m_mapValues.begin();
	for (; mapit != from.m_mapValues.end(); mapit++)
	{
		IniSection mapEmpty;
		IniFile::iterator /*mapfound = m_mapValues.find(mapit->first);
						  if(mapfound == m_mapValues.end())*/
						  // new empty map is added only if such a key does not exist
						  mapfound = m_mapValues.insert(std::pair<std::string, IniSection >(mapit->first, mapEmpty)).first;
		IniSection::const_iterator it = mapit->second.begin();
		for (; it != mapit->second.end(); it++)
		{
			IniSection::iterator found = mapfound->second.find(it->first);
			// if bCopyValues is false and value already exists, then we does not have to change it
			if (bCopyValues || found == mapfound->second.end())
				mapfound->second[it->first] = bCopyValues ? it->second : "";
		}
	}
}

void IniManager::UpdateValues(const IniManager& from)
{
	IniFile::const_iterator mapit = from.m_mapValues.begin();
	for (; mapit != from.m_mapValues.end(); mapit++)
	{
		IniSection mapEmpty;
		IniFile::iterator mapfound = m_mapValues.find(mapit->first);
		if (mapfound != m_mapValues.end())
		{
			IniSection::const_iterator it = mapit->second.begin();
			for (; it != mapit->second.end(); it++)
			{
				IniSection::iterator found = mapfound->second.find(it->first);
				if (found != mapfound->second.end())
					mapfound->second[it->first] = it->second;
			}
		}
	}
}

void IniManager::UpdateValueLists(const IniManager& from)
{
	IniFile::const_iterator mapit = from.m_mapValues.begin();
	for (; mapit != from.m_mapValues.end(); mapit++)
	{
		IniSection mapEmpty;
		IniFile::iterator mapfound = m_mapValues.find(mapit->first);
		if (mapfound != m_mapValues.end())
		{
			IniSection::const_iterator it = mapit->second.begin();
			for (; it != mapit->second.end(); it++)
			{
				IniSection::iterator found = mapfound->second.find(it->first);
				if (found != mapfound->second.end())
				{
					std::string sVal = it->second;
					std::string sList = found->second;
					if (!sVal.empty() && sVal.find(';') == std::string::npos &&
						sList.find(';') != std::string::npos)
					{
#ifndef __linux__

						std::vector<std::string> & vecOptions = Tokenize(sList.c_str(), ";");
#else
						std::vector<std::string> vecOptions = Tokenize(sList.c_str(), ";");
#endif
						unsigned i;
						for (i = 0; i < vecOptions.size(); i++)
							if (vecOptions[i] == sVal)
								break;
						if (i == vecOptions.size())
						{
							sList += ';' + sVal;
							mapfound->second[it->first] = sList;
						}
					}
				}
			}
		}
	}
}

std::string IniManager::FindValue(const char* szSection, const char* szName) const
{
	std::string str, sfind(szSection);
	std::transform(sfind.begin(), sfind.end(), sfind.begin(), ::tolower);
	IniFile::const_iterator mapit = m_mapValues.find(sfind);
	if (mapit != m_mapValues.end())
	{
		sfind = szName;
		transform(sfind.begin(), sfind.end(), sfind.begin(), ::tolower);
		IniSection::const_iterator it = mapit->second.find(sfind);
		if (it != mapit->second.end())
			str = it->second;
	}
	return str;
}

std::string IniManager::GetValue(const IniSection& section, const char* szName) const
{
	std::string str, sfind(szName);
	std::transform(sfind.begin(), sfind.end(), sfind.begin(), ::tolower);
	IniSection::const_iterator it = section.find(sfind);
	if (it != section.end())
		str = it->second;
	return str;
}

void IniManager::AddValue(const char* szSection, const char* szName, const char* szValue)
{
	std::string ss(szSection), sn(szName);
	std::transform(ss.begin(), ss.end(), ss.begin(), ::tolower);
	std::transform(sn.begin(), sn.end(), sn.begin(), ::tolower);
	m_mapValues[ss][sn] = szValue;
};

bool IniManager::IsSection(const char* szSection) const
{
	std::string sfind(szSection);
	transform(sfind.begin(), sfind.end(), sfind.begin(), ::tolower);
	return (m_mapValues.find(sfind) != m_mapValues.end());
}

const IniManager::IniSection& IniManager::GetSection(const char* szSection) const
{
	std::string sfind(szSection);
	transform(sfind.begin(), sfind.end(), sfind.begin(), ::tolower);
	return m_mapValues.find(sfind)->second;
}

std::string IniManager::GetString(const char* szSection, const char* szName, const char* szDefault) const
{
	std::string stmp;
	stmp = FindValue(szSection, szName);
	if (stmp.empty())
		stmp = szDefault;
	return stmp;
}

int IniManager::GetInt(const char* section, const char* name, int nDefault) const
{
	int var = nDefault;
	std::string stmp = FindValue(section, name);
	if (!stmp.empty())
	{
		sscanf(stmp.c_str(), "%d", &var);
		/*var=atoi(stmp.c_str());*/
		if (var == 0 && (stmp[1] == 'x' || stmp[1] == 'X'))
			sscanf(stmp.c_str(), "0x%x", &var);
	}
	return var;
}

int IniManager::GetInt(const IniSection& section, const char* name, int nDefault) const
{
	int var = nDefault;
	std::string stmp = GetValue(section, name);
	if (!stmp.empty())
	{
		sscanf(stmp.c_str(), "%d", &var);
		/*var=atoi(stmp.c_str());*/
		if (var == 0 && (stmp[1] == 'x' || stmp[1] == 'X'))
			sscanf(stmp.c_str(), "0x%x", &var);
	}
	return var;
}

std::vector<std::string> Tokenize(const char* szList, const char* szDelim)
{
	std::vector<std::string> ret;
	if (!strlen(szList))
		return ret;
	char * szBuf = new char[strlen(szList) + 1];
	char * szTok = szBuf;

	strcpy(szBuf, szList);
	szTok = strtok(szTok, szDelim);
	while (szTok)
	{
		ret.push_back(szTok);
		szTok = strtok(NULL, szDelim);
	}

	delete[] szBuf;
	return ret;
}
