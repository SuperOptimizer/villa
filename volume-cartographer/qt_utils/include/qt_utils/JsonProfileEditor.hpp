#pragma once

#include <QGroupBox>
#include <QJsonObject>
#include <QVector>

#include <optional>

class QComboBox;
class QLabel;
class QPlainTextEdit;
class QPushButton;

namespace qt_utils
{

/// A QGroupBox containing a dropdown profile selector and a JSON text editor.
///
/// Features:
///   - Built-in profiles (read-only) and a "Custom" profile (always editable)
///   - User-created profiles saved via QSettings (editable, persistent)
///   - Live JSON validation with error display
///   - Save As / Rename / Delete buttons for user profiles
///
/// The `settingsGroup` parameter controls the QSettings key under which user
/// profiles are stored, allowing multiple independent editors in the same app.
class JsonProfileEditor : public QGroupBox
{
    Q_OBJECT

public:
    /// A profile definition.
    struct Profile {
        QString id;          ///< Unique identifier (e.g. "default", "custom", "user_3")
        QString label;       ///< Display name shown in the combo box
        QString jsonText;    ///< The JSON text content
        bool editable{false};///< Whether the text editor is writable for this profile
    };

    /// @param title          Group box title.
    /// @param settingsGroup  QSettings group key for user profile persistence.
    /// @param parent         Parent widget.
    explicit JsonProfileEditor(
        const QString& title,
        const QString& settingsGroup = QStringLiteral("user_profiles"),
        QWidget* parent = nullptr);

    // --- Configuration ---

    /// Set an optional description label above the profile selector.
    void setDescription(const QString& text);

    /// Set placeholder text for the JSON text editor.
    void setPlaceholderText(const QString& text);

    /// Set a tooltip on the JSON text editor.
    void setTextToolTip(const QString& text);

    /// Provide the set of available profiles and select a default.
    /// A "Custom" profile is automatically prepended if not already present.
    void setProfiles(const QVector<Profile>& profiles, const QString& defaultProfileId = {});

    /// Optional: set a custom path for the QSettings INI file used by
    /// UserProfileStore. If empty (default), uses QCoreApplication defaults.
    void setSettingsPath(const QString& path);

    // --- Profile access ---

    /// Programmatically switch to a profile by ID.
    void setProfile(const QString& profileId, bool fromUi = false);

    /// Return the current profile ID.
    [[nodiscard]] auto profile() const -> QString;

    /// Return the custom (free-form) text buffer.
    [[nodiscard]] auto customText() const -> QString;

    /// Set the custom text buffer. Only applies when the "custom" profile is active.
    void setCustomText(const QString& text);

    /// Return the current text in the editor (regardless of profile).
    [[nodiscard]] auto currentText() const -> QString;

    // --- Validation ---

    /// Parse the current text as a JSON object.
    /// On parse error, returns std::nullopt and (if non-null) fills *error.
    [[nodiscard]] auto jsonObject(QString* error = nullptr) const
        -> std::optional<QJsonObject>;

    /// Returns true if the current text parses as valid JSON (or is empty).
    [[nodiscard]] auto isValid() const -> bool;

    /// Return the current validation error text (empty if valid).
    [[nodiscard]] auto errorText() const -> QString;

    /// Reload user profiles from QSettings into the combo box.
    void loadUserProfiles();

signals:
    /// Emitted when the selected profile changes.
    void profileChanged(const QString& profileId);

    /// Emitted when the editor text changes (by user or programmatically).
    void textChanged();

    /// Emitted after a user profile is saved, renamed, or deleted.
    void userProfilesChanged();

private:
    void applyProfile(const QString& profileId, bool fromUi);
    void handleTextEdited();
    void validateCurrentText();
    auto findProfile(const QString& id) const -> const Profile*;
    void updateStatusLabel();

    void onSaveAs();
    void onRename();
    void onDelete();
    void updateUserProfileButtons();
    [[nodiscard]] auto isUserProfile() const -> bool;
    void rebuildCombo();

    QLabel* description_{nullptr};
    QComboBox* profileCombo_{nullptr};
    QPlainTextEdit* textEdit_{nullptr};
    QLabel* statusLabel_{nullptr};

    QPushButton* saveAsBtn_{nullptr};
    QPushButton* renameBtn_{nullptr};
    QPushButton* deleteBtn_{nullptr};

    QVector<Profile> profiles_;
    QString profileId_{QStringLiteral("custom")};
    QString customText_;
    QString errorText_;
    QString settingsGroup_;
    QString settingsPath_;
    bool updatingProgrammatically_{false};
};

}  // namespace qt_utils
