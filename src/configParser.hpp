#pragma once

#include <vector>
#include <map>
#include <string>
#include <ostream>

//
// simple class for reading/writing ini-file and storing data 

class ConfigManager
{
public:
	typedef std::map< std::string, std::string > ConfigSection;
	typedef std::map< std::string, ConfigSection > ConfigFile;

	ConfigManager(std::string szFileName) : m_sFileName(szFileName) { };
	const char* GetIniFile() { return m_sFileName.c_str(); };
	void ReadIniFile(/*const char* szFileName*/);

	// get reference to values, organized in hierarchical map
	const ConfigFile& GetAllValues() const { return m_mapValues; };
	// add value with given section and field name
	void AddValue(const char* szSection, const char* szName, const char* szValue);
	// find value, return empty string if value could not be found
	std::string FindValue(const char* szSection, const char* szName) const;
	std::string GetValue(const ConfigSection& section, const char* szName) const;
	// find section, return true if section is found
	bool IsSection(const char* szSection) const;
	// get section; will throw exception if section is not found
	const ConfigSection& GetSection(const char* szSection) const;
	// 
	// find integer value if it exists, otherwise return default (understands decimal and hex values, prefixed with 0x)
	int GetInt(const char* section, const char* name, int nDefault) const;
	// 
	int GetInt(const ConfigSection& section, const char* name, int nDefault) const;
	// find string value if it exists, otherwise return default
	std::string GetString(const char* szSection, const char* szName, const char* szDefault) const;
	// remove all values
	void Clear() { m_mapValues.clear(); };
	// remove values from section
	void ClearSection(const char* szSection);
	// copy data to this ini-file from other; if value exists, it is updated (depending on bCopyValues flag), if it's not, it is added
	// if bCopyValues is true, then values are updated, otherwise only descriptions are copied and values are empty
	// if bClear is true, then previous contents are removed before copying
	void Copy(const IniManager& from, bool bCopyValues = true, bool bClear = false);
	// the same as Copy with bCopyValues == true and bClear == false,
	// but only existing values are refreshed
	void UpdateValues(const IniManager& from);
	// update values of DB description files (each value in ini-file is a list of DB field values, divided by semicolon)
	void UpdateValueLists(const IniManager& from);
	// output ini-file to stream
	void Write(std::ostream& to) const;

protected:
	IniFile m_mapValues; // map sections to maps of entries to values
	std::string m_sFileName;
};

//////////////////////////////////////////////////////////////////
// Helper function(s)

std::vector<std::string> Tokenize(const char* szList, const char* szDelim);
