#pragma once

#include <QSettings>
#include <QString>
#include <QVariantMap>
#include <QVector>

namespace qt_utils
{

/// A header-only, QSettings-backed profile CRUD store.
///
/// Profiles are organized by "group" (an arbitrary string key, e.g. "segmentation_params").
/// Each profile within a group has a numeric storage ID, a name, and arbitrary string data.
///
/// Storage layout in QSettings:
///   <group>/next_id = <int>
///   <group>/<id>/name = <string>
///   <group>/<id>/data = <string>
///
/// By default, uses QSettings with QCoreApplication's organizationName/applicationName.
/// Callers may supply a custom settings file path instead.
namespace UserProfileStore
{

/// A single stored user profile.
struct Profile {
    int storageId{0};
    QString name;
    QString data;
};

/// Build the profile ID string from a numeric storage ID.
/// Returns "user_<storageId>", e.g. "user_3".
inline auto profileIdFromStorageId(int storageId) -> QString
{
    return QStringLiteral("user_%1").arg(storageId);
}

/// Extract the numeric storage ID from a profile ID string.
/// Returns -1 if the string does not match the "user_<N>" pattern.
inline auto storageIdFromProfileId(const QString& profileId) -> int
{
    if (!profileId.startsWith(QStringLiteral("user_"))) {
        return -1;
    }
    bool ok = false;
    const int id = profileId.mid(5).toInt(&ok);
    return ok ? id : -1;
}

/// Test whether a profile ID string refers to a user-created profile.
inline auto isUserProfileId(const QString& profileId) -> bool
{
    return storageIdFromProfileId(profileId) > 0;
}

/// Load all profiles within a group, ordered by storage ID.
/// @param group  The settings group key.
/// @param settingsPath  Optional path for a custom QSettings INI file.
///                      If empty, uses default QSettings.
inline auto loadAll(
    const QString& group,
    const QString& settingsPath = {}) -> QVector<Profile>
{
    QSettings settings =
        settingsPath.isEmpty()
            ? QSettings()
            : QSettings(settingsPath, QSettings::IniFormat);

    QVector<Profile> result;

    settings.beginGroup(group);
    const int nextId = settings.value(QStringLiteral("next_id"), 1).toInt();

    for (int id = 1; id < nextId; ++id) {
        settings.beginGroup(QString::number(id));
        const QString name = settings.value(QStringLiteral("name")).toString();
        if (!name.isEmpty()) {
            Profile profile;
            profile.storageId = id;
            profile.name = name;
            profile.data = settings.value(QStringLiteral("data")).toString();
            result.append(profile);
        }
        settings.endGroup();
    }

    settings.endGroup();
    return result;
}

/// Save a new profile and return its storage ID.
inline auto save(
    const QString& group,
    const QString& name,
    const QString& data,
    const QString& settingsPath = {}) -> int
{
    QSettings settings =
        settingsPath.isEmpty()
            ? QSettings()
            : QSettings(settingsPath, QSettings::IniFormat);

    settings.beginGroup(group);
    const int id = settings.value(QStringLiteral("next_id"), 1).toInt();
    settings.setValue(QStringLiteral("next_id"), id + 1);

    settings.beginGroup(QString::number(id));
    settings.setValue(QStringLiteral("name"), name);
    settings.setValue(QStringLiteral("data"), data);
    settings.endGroup();

    settings.endGroup();
    return id;
}

/// Update an existing profile's name and data.
inline void update(
    const QString& group,
    int storageId,
    const QString& name,
    const QString& data,
    const QString& settingsPath = {})
{
    QSettings settings =
        settingsPath.isEmpty()
            ? QSettings()
            : QSettings(settingsPath, QSettings::IniFormat);

    settings.beginGroup(group);
    settings.beginGroup(QString::number(storageId));
    settings.setValue(QStringLiteral("name"), name);
    settings.setValue(QStringLiteral("data"), data);
    settings.endGroup();
    settings.endGroup();
}

/// Delete a profile by storage ID.
inline void remove(
    const QString& group,
    int storageId,
    const QString& settingsPath = {})
{
    QSettings settings =
        settingsPath.isEmpty()
            ? QSettings()
            : QSettings(settingsPath, QSettings::IniFormat);

    settings.beginGroup(group);
    settings.remove(QString::number(storageId));
    settings.endGroup();
}

/// List all profile names within a group.
inline auto listProfiles(
    const QString& group,
    const QString& settingsPath = {}) -> QStringList
{
    const auto profiles = loadAll(group, settingsPath);
    QStringList names;
    names.reserve(profiles.size());
    for (const auto& p : profiles) {
        names.append(p.name);
    }
    return names;
}

/// Load a single profile by its storage ID. Returns a default-constructed
/// Profile (storageId == 0) if not found.
inline auto loadProfile(
    const QString& group,
    int storageId,
    const QString& settingsPath = {}) -> Profile
{
    QSettings settings =
        settingsPath.isEmpty()
            ? QSettings()
            : QSettings(settingsPath, QSettings::IniFormat);

    settings.beginGroup(group);
    settings.beginGroup(QString::number(storageId));
    const QString name = settings.value(QStringLiteral("name")).toString();
    Profile profile;
    if (!name.isEmpty()) {
        profile.storageId = storageId;
        profile.name = name;
        profile.data = settings.value(QStringLiteral("data")).toString();
    }
    settings.endGroup();
    settings.endGroup();
    return profile;
}

/// Check whether a profile with the given storage ID exists.
inline auto profileExists(
    const QString& group,
    int storageId,
    const QString& settingsPath = {}) -> bool
{
    QSettings settings =
        settingsPath.isEmpty()
            ? QSettings()
            : QSettings(settingsPath, QSettings::IniFormat);

    settings.beginGroup(group);
    settings.beginGroup(QString::number(storageId));
    const bool exists = !settings.value(QStringLiteral("name")).toString().isEmpty();
    settings.endGroup();
    settings.endGroup();
    return exists;
}

}  // namespace UserProfileStore

}  // namespace qt_utils
